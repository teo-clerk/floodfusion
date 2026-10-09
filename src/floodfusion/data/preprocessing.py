r"""Sentinel-1 pre-processing: calibration, noise removal, filtering, terrain correction.

The chain, in order (``docs/04`` section 8). **Order matters; several steps do not
commute.**

::

     1. Read GRD SAFE                    measurement TIFF + annotation XML
     2. Apply orbit file                 precise orbits; latency of hours to ~20 days
     3. Remove GRD border noise          BEFORE any statistic
     4. Remove thermal noise             in LINEAR power
     5. Radiometric calibration          DN^2 / A^2  ->  beta0 / sigma0 / gamma0
     6. (optional) Speckle filter        in LINEAR power
     7. Range-Doppler terrain correction output on the analysis grid
     8. Radiometric terrain correction   -> gamma0_T
     9. Convert to dB                    LAST
    10. Store as int16 (dB x 100)        halves disk vs float32

Why that order: 3 first because border artefacts corrupt every later statistic; 4 before
5 because noise subtraction is defined on power at the pre-calibration scale; 6 before 7
because filtering assumes the speckle model with a known number of looks, and geocoding
correlates neighbouring pixels so the effective L is no longer what the filter assumes;
9 last because of the Jensen bias below.

References
----------
``docs/04_data_access_and_sentinel1_preprocessing.md`` sections 3--6, 8.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

__all__ = [
    "DB_NODATA",
    "calibrate",
    "db_to_linear",
    "equivalent_number_of_looks",
    "interpolate_lut",
    "linear_to_db",
    "refined_lee",
    "remove_thermal_noise",
    "terrain_correct",
    "thermal_noise_report",
]

LOGGER = logging.getLogger(__name__)

#: The nodata sentinel expressed in dB: int16 -32768 / db_scale 100 (docs/04 section 8.1).
#: Far outside the physical range of gamma0 (~[-35, +10] dB), so it can never be mistaken
#: for a measurement once it is routed into the validity mask.
DB_NODATA: float = -327.68

_CALIBRATION_QUANTITIES = {
    "sigma": "sigmaNought", "sigmanought": "sigmaNought",
    "beta": "betaNought", "betanought": "betaNought",
    "gamma": "gamma",
    "dn": "dn",
}


def interpolate_lut(
    lut: np.ndarray,
    shape: tuple[int, int],
    *,
    lut_lines: Sequence[float] | np.ndarray | None = None,
    lut_pixels: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Bilinearly interpolate a coarse calibration/noise LUT to full raster resolution.

    Sentinel-1 calibration and noise vectors are given at a sparse lattice of azimuth
    lines x range pixels (``docs/04`` section 3.2). ``lut`` is ``(n_lines, n_pixels)``
    sampled at ``lut_lines`` x ``lut_pixels``; the result is ``shape``. Bilinear and
    separable, edge values held constant beyond the lattice. A LUT that already has the
    raster's shape is returned unchanged.
    """
    lut = np.asarray(lut, dtype=np.float64)
    if lut.shape == tuple(shape):
        return lut
    if lut_lines is None or lut_pixels is None:
        raise ValueError(
            f"LUT shape {lut.shape} differs from raster {shape}: pass lut_lines and "
            "lut_pixels (the azimuth line / range pixel of every LUT sample)"
        )
    lines = np.asarray(lut_lines, dtype=np.float64)
    pixels = np.asarray(lut_pixels, dtype=np.float64)
    if lut.shape != (lines.size, pixels.size):
        raise ValueError(f"LUT shape {lut.shape} != ({lines.size}, {pixels.size})")
    cols = np.arange(shape[1], dtype=np.float64)
    rows = np.arange(shape[0], dtype=np.float64)
    along_range = np.stack([np.interp(cols, pixels, row) for row in lut])  # (n_lines, W)
    out = np.empty(shape, dtype=np.float64)
    for j in range(shape[1]):
        out[:, j] = np.interp(rows, lines, along_range[:, j])
    return out


def calibrate(
    dn: np.ndarray,
    lut: np.ndarray,
    quantity: str = "gamma",
    *,
    lut_lines: Sequence[float] | np.ndarray | None = None,
    lut_pixels: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    r"""Convert digital numbers to a calibrated backscatter quantity.

    .. math:: \text{value}(i,j) = \frac{|\mathrm{DN}(i,j)|^2}{A(i,j)^2}

    with :math:`A` the LUT for the desired quantity -- ``sigmaNought``, ``betaNought``,
    ``gamma`` or ``dn`` -- read from ``annotation/calibration/calibration-*.xml``.

    Notes
    -----
    **Note the squares.** DN is an amplitude; the calibrated quantities are powers.
    Forgetting the square is a factor-of-two error in dB that produces values which look
    almost plausible, which is what makes it dangerous.

    The LUTs are on a **coarse grid** and must be bilinearly interpolated to full
    resolution first.

    **The three normalisations** differ in which area they divide by:

    - :math:`\beta^0` -- slant-range area; the natural radar measurement.
    - :math:`\sigma^0` -- horizontal ground area; the conventional reporting quantity.
    - :math:`\gamma^0` -- area perpendicular to the look direction; flattest over terrain.

    For flat terrain, :math:`\sigma^0 = \beta^0\sin\theta` and
    :math:`\gamma^0 = \sigma^0/\cos\theta = \beta^0\tan\theta`.

    **This project uses** :math:`\gamma^0`, because :math:`\sigma^0` retains a systematic
    incidence-angle dependence: across an IW swath (29 to 46 degrees) that is a gradient
    of several dB that the network would have to learn to ignore, and which invalidates a
    global threshold. Over terrain the correct angle is the **local** incidence angle of
    ``docs/03`` section 5, not the ellipsoid angle -- that distinction is exactly the
    difference between geocoding and radiometric terrain correction.

    TODO(student)
    -------------
    1. Interpolate the LUT to full resolution.
    2. Apply the formula. Assert the output is non-negative.
    3. Compare a 256x256 window against a toolbox result and report the maximum absolute
       dB difference (``docs/04`` exercise 1). Discrepancies are usually LUT
       interpolation, then the missing square.
    """
    key = quantity.replace("_", "").lower()
    if key not in _CALIBRATION_QUANTITIES:
        raise ValueError(f"unknown calibration quantity {quantity!r}; use sigma, beta, gamma or dn")
    dn = np.asarray(dn, dtype=np.float64)
    a = interpolate_lut(lut, dn.shape, lut_lines=lut_lines, lut_pixels=lut_pixels)
    if np.any(a <= 0.0):
        raise ValueError("calibration LUT must be strictly positive")
    value = np.abs(dn) ** 2 / a**2  # DN is an AMPLITUDE: square it (docs/04 3.2)
    if np.any(value < 0.0):  # pragma: no cover - impossible by construction
        raise AssertionError("calibrated power is negative")
    return value


def remove_thermal_noise(
    power: np.ndarray,
    noise_lut: np.ndarray,
    *,
    lut_lines: Sequence[float] | np.ndarray | None = None,
    lut_pixels: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    r"""Subtract the receiver noise floor, in linear power.

    .. math:: \sigma^0_{\text{corrected}} = \sigma^0_{\text{measured}} - \sigma^0_{\text{noise}}

    Notes
    -----
    Two consequences specific to flood mapping:

    1. **The result can be negative** where the true signal is at or below the noise
       floor, and :math:`\log` is then undefined. Do not silently clip to a small
       positive constant **without recording how many pixels you clipped** -- that count
       is a direct measurement of how much of your scene is noise-limited, and a large
       fraction over water is expected and informative.
    2. This is exactly the **VH-over-water regime**: calm water in VH sits at or below
       the NESZ of -22 to -25 dB, so before denoising you partly measure the instrument
       and after denoising you difference two similar numbers. It is why VH-only
       thresholding is fragile (``docs/01`` section 3.3).

    Denoising also removes the **scalloping** -- periodic brightness variation across IW
    sub-swaths from the TOPSAR antenna pattern -- which otherwise appears as vertical
    banding that a CNN will happily learn as a feature.

    TODO(student)
    -------------
    1. Interpolate the noise LUT to full resolution.
    2. Subtract in linear power.
    3. Return the corrected array **and** the count of non-positive pixels. Log it.
    """
    power = np.asarray(power, dtype=np.float64)
    noise = interpolate_lut(noise_lut, power.shape, lut_lines=lut_lines, lut_pixels=lut_pixels)
    corrected = power - noise
    report = thermal_noise_report(corrected)
    LOGGER.info(
        "thermal noise removal: %d of %d pixels (%.2f %%) are at or below the noise floor",
        report["non_positive"], report["total"], 100.0 * report["non_positive_fraction"],
    )
    return corrected


def thermal_noise_report(corrected: np.ndarray) -> dict[str, float]:
    """Count noise-limited pixels after denoising (``docs/04`` section 4.2, point 1).

    The fraction is a direct measurement of how much of the scene is below the NESZ — a
    large value over water in VH is expected and informative, and it is recorded rather
    than hidden by silently clipping to a small positive constant.
    """
    corrected = np.asarray(corrected)
    finite = np.isfinite(corrected)
    non_positive = int(np.count_nonzero(finite & (corrected <= 0.0)))
    total = int(np.count_nonzero(finite))
    return {
        "non_positive": non_positive,
        "total": total,
        "non_positive_fraction": non_positive / total if total else float("nan"),
    }


def linear_to_db(power: np.ndarray) -> np.ndarray:
    r"""Convert linear power to decibels: :math:`10\log_{10}(\cdot)`.

    Notes
    -----
    **Convert last, and only for storage, display and thresholding.** All filtering,
    multi-looking and spatial averaging happens in linear power, because Jensen's
    inequality makes dB averaging systematically **low**:

    .. math:: b_{\mathrm{dB}}(L) = 4.343\,[\psi(L) - \ln L]

    which is :math:`-2.51` dB at :math:`L=1`, :math:`-0.51` dB at :math:`L=4.4` (IW
    GRDH), and :math:`-0.11` dB at :math:`L=20`. A -0.5 dB systematic offset is
    comparable to the gap between a correct and an incorrect flood threshold -- and it
    is **not constant**, because it depends on the effective number of looks, which
    varies with your filter and window size. So it is a spatially varying bias.

    TODO(student)
    -------------
    Implement, handling non-positive input explicitly (return the nodata sentinel, not
    ``-inf``), and add the test asserting mean-of-window equals the linear-domain mean.
    """
    power = np.asarray(power, dtype=np.float64)
    out = np.full(power.shape, DB_NODATA, dtype=np.float64)
    ok = np.isfinite(power) & (power > 0.0)
    out[ok] = 10.0 * np.log10(power[ok])
    return out


def db_to_linear(db: np.ndarray) -> np.ndarray:
    r"""Convert decibels to linear power: :math:`10^{(\cdot)/10}`."""
    db = np.asarray(db, dtype=np.float64)
    out = np.power(10.0, db / 10.0)
    out[~np.isfinite(db) | (db <= DB_NODATA)] = np.nan  # the sentinel is not a measurement
    return out


def refined_lee(intensity: np.ndarray, looks: float, window: int = 7) -> np.ndarray:
    r"""Refined Lee speckle filter.

    Notes
    -----
    **The model.** Speckle is multiplicative: :math:`I = R \cdot F` with
    :math:`\mathbb{E}[F]=1` and :math:`\operatorname{Var}[F] = 1/L`. It is not
    measurement noise -- it is coherent interference of sub-resolution scatterers, and it
    is deterministic given the scene; it is treated statistically because the arrangement
    is unknown.

    **Plain Lee** is the linear MMSE estimator of :math:`R` in a local window. With
    :math:`\bar{I}` the local mean, :math:`C_I = \sigma_I/\bar{I}` the local coefficient
    of variation and :math:`C_u = 1/\sqrt{L}` that of pure speckle:

    .. math:: \hat{R} = \bar{I} + W(I - \bar{I}), \qquad W = 1 - \frac{C_u^2}{C_I^2}

    (clamped to :math:`[0,1]`). In a homogeneous region :math:`C_I \approx C_u`,
    :math:`W \approx 0`, and the output is the local mean -- maximum smoothing. At an
    edge or point target :math:`C_I \gg C_u`, :math:`W \to 1`, and the pixel passes
    through unchanged.

    **What "refined" refines.** A square window at an edge straddles two populations,
    inflating :math:`C_I`, driving :math:`W \to 1`, and switching the filter off in a
    band around every edge -- precisely where flood boundaries are. Refined Lee selects
    an edge-aligned sub-window on the *homogeneous side*::

        1. Compute the 3x3 sub-block means of the 7x7 window.
        2. Estimate edge direction from gradients of that 3x3 array
           (4 candidate directions -> 8 oriented half-windows).
        3. Select the oriented sub-window on ONE side of the edge.
        4. Compute I_bar and C_I within THAT sub-window only.
        5. Apply the Lee weight above.

    **Should you filter at all?** A real design question. Filter for the ``P1-M2``
    threshold baseline, where a single-pixel decision at
    :math:`\sigma_{\mathrm{dB}} = 2.17` dB is hopeless. Feed the **network** unfiltered
    :math:`\gamma^0` and let it learn its own spatial aggregation -- then *test* the
    assumption as an extra ablation row (``docs/04`` section 6.4).

    TODO(student)
    -------------
    1. Operate on **linear intensity**, never dB.
    2. Implement plain Lee first and plot :math:`W` across a synthetic edge.
    3. Add the refinement; show the edge-band artefact disappears.
    4. Report ENL in a homogeneous patch before and after -- that is the measurement that
       tells you the effective :math:`L` your change-detection threshold should use.
    """
    from scipy import ndimage

    if window < 5 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 5")
    if looks <= 0:
        raise ValueError("looks must be positive")
    image = np.asarray(intensity, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("refined_lee expects a 2-D intensity array")
    valid = np.isfinite(image) & (image > 0.0)
    x = np.where(valid, image, 0.0)
    w = valid.astype(np.float64)
    half = window // 2

    # 1. 3x3 array of sub-block means, sub-blocks centred at offsets {-step, 0, +step}.
    sub = max(1, (window // 3) | 1)  # odd sub-block size: 3 for a 7x7 window
    step = (window - sub) // 2
    sum_sub = ndimage.uniform_filter(x, size=sub, mode="reflect")
    cnt_sub = ndimage.uniform_filter(w, size=sub, mode="reflect")
    mean_sub = np.divide(sum_sub, cnt_sub, out=np.zeros_like(sum_sub), where=cnt_sub > 0)
    padded = np.pad(mean_sub, step, mode="reflect")
    h, wd = image.shape

    def block(a: int, b: int) -> np.ndarray:  # a, b in {0, 1, 2}
        r0, c0 = a * step, b * step
        return padded[r0:r0 + h, c0:c0 + wd]

    m = [[block(a, b) for b in range(3)] for a in range(3)]
    left = m[0][0] + m[1][0] + m[2][0]
    right = m[0][2] + m[1][2] + m[2][2]
    top = m[0][0] + m[0][1] + m[0][2]
    bottom = m[2][0] + m[2][1] + m[2][2]
    upper_right = m[0][1] + m[0][2] + m[1][2]
    lower_left = m[1][0] + m[2][0] + m[2][1]
    upper_left = m[0][0] + m[0][1] + m[1][0]
    lower_right = m[1][2] + m[2][1] + m[2][2]

    # 2. Edge direction = the strongest of four gradients of the 3x3 mean array.
    pairs = (
        (left, right),             # vertical edge   -> left / right half-window
        (top, bottom),             # horizontal edge -> top / bottom
        (upper_right, lower_left), # edge along the main diagonal
        (upper_left, lower_right), # edge along the anti-diagonal
    )
    gradients = np.stack([np.abs(b - a) for a, b in pairs])
    direction = np.argmax(gradients, axis=0)

    # 3. The homogeneous side: the half whose sub-block mean is closer to the centre.
    centre = m[1][1]
    side = np.zeros(image.shape, dtype=np.int64)
    for d, (a, b) in enumerate(pairs):
        pick_b = np.abs(b / 3.0 - centre) < np.abs(a / 3.0 - centre)
        side = np.where(direction == d, pick_b.astype(np.int64), side)
    choice = 2 * direction + side  # 0..7

    # 4. Local statistics inside each of the 8 oriented half-windows (incl. the centre line).
    ii, jj = np.mgrid[0:window, 0:window]
    masks = (
        jj <= half, jj >= half,                  # left, right
        ii <= half, ii >= half,                  # top, bottom
        jj >= ii, jj <= ii,                      # upper-right, lower-left triangles
        ii + jj <= 2 * half, ii + jj >= 2 * half,  # upper-left, lower-right triangles
    )
    mean = np.zeros_like(x)
    var = np.zeros_like(x)
    x2 = x * x
    for k, mask in enumerate(masks):
        kernel = mask.astype(np.float64)
        n = ndimage.correlate(w, kernel, mode="reflect")
        s1 = ndimage.correlate(x, kernel, mode="reflect")
        s2 = ndimage.correlate(x2, kernel, mode="reflect")
        mk = np.divide(s1, n, out=np.zeros_like(s1), where=n > 0)
        vk = np.divide(s2, n, out=np.zeros_like(s2), where=n > 0) - mk**2
        sel = choice == k
        mean[sel] = mk[sel]
        var[sel] = np.maximum(vk[sel], 0.0)

    # 5. Lee MMSE weight with the oriented statistics, clamped to [0, 1].
    cu2 = 1.0 / looks
    ci2 = np.divide(var, mean**2, out=np.full_like(var, np.inf), where=mean > 0)
    weight = np.clip(1.0 - cu2 / ci2, 0.0, 1.0)
    out = mean + weight * (x - mean)
    out[~valid] = image[~valid]  # nodata / non-positive pixels pass through untouched
    return out


def equivalent_number_of_looks(intensity: np.ndarray) -> float:
    """ENL = (mean / std)^2 over a homogeneous patch of LINEAR intensity.

    The measurement that sets the change-detection threshold: after filtering, the
    effective L is no longer 4.4 (``docs/04`` section 6.4, ``docs/01`` section 10.2).
    """
    values = np.asarray(intensity, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        raise ValueError("need at least two valid pixels")
    std = values.std()
    return float("inf") if std == 0.0 else float((values.mean() / std) ** 2)


def terrain_correct(*args: object, **kwargs: object) -> np.ndarray:
    r"""Range-Doppler terrain correction, backward-geocoded onto the analysis grid.

    Implemented in :func:`floodfusion.data.geocoding.terrain_correct` (this name is kept
    here because ``docs/04`` section 8 lists it as step 7-8 of the pre-processing chain);
    see that module for the zero-Doppler solve, backward geocoding in linear power and the
    area-based RTC of Small (2011):

    .. math:: \gamma^0_T = \beta^0 \cdot \frac{A_\beta}{A_\gamma}

    **And the result is still untrustworthy in layover.** Where
    :math:`\theta_{\mathrm{loc}} \le 0`, returns from multiple distinct ground patches are
    summed in a single range bin; no division by an area recovers information destroyed in
    the receiver. That is ``P1-M1``'s comprehension question.

    Production ingest uses the pre-processed Planetary Computer ``sentinel-1-rtc``
    collection (Route B, ``docs/04`` section 9), which already ran these steps.
    """
    from floodfusion.data.geocoding import terrain_correct as _terrain_correct

    return _terrain_correct(*args, **kwargs)  # type: ignore[arg-type]
