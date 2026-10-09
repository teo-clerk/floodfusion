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

import numpy as np

__all__ = [
    "calibrate",
    "db_to_linear",
    "linear_to_db",
    "refined_lee",
    "remove_thermal_noise",
    "terrain_correct",
]


def calibrate(dn: np.ndarray, lut: np.ndarray, quantity: str = "gamma") -> np.ndarray:
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
    raise NotImplementedError("Implement calibration. See docs/04 section 3.2.")


def remove_thermal_noise(power: np.ndarray, noise_lut: np.ndarray) -> np.ndarray:
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
    raise NotImplementedError("Implement noise removal. See docs/04 section 4.2.")


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
    raise NotImplementedError("Implement the dB conversion. See docs/04 section 3.4.")


def db_to_linear(db: np.ndarray) -> np.ndarray:
    r"""Convert decibels to linear power: :math:`10^{(\cdot)/10}`."""
    raise NotImplementedError("Implement the inverse conversion. See docs/04 section 3.4.")


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
    raise NotImplementedError("Implement Refined Lee. See docs/04 section 6.")


def terrain_correct(*args: object, **kwargs: object) -> np.ndarray:
    r"""Range-Doppler terrain correction, backward-geocoded onto the analysis grid.

    Notes
    -----
    Sentinel-1 products are processed to **zero-Doppler** geometry. For a ground target
    at ECEF position :math:`\mathbf{P}` and a satellite at :math:`\mathbf{S}(t)` with
    velocity :math:`\mathbf{V}(t)`, the imaging time :math:`t^\star` satisfies

    .. math:: \mathbf{V}(t^\star)\cdot[\mathbf{P} - \mathbf{S}(t^\star)] = 0

    -- geometrically, closest approach -- and the slant range is
    :math:`R = \|\mathbf{P} - \mathbf{S}(t^\star)\|`.

    **Backward geocoding**: iterate over the **output** grid, and for each cell solve for
    :math:`(t^\star, R)` and interpolate the source image there::

        FOR each cell g of the OUTPUT grid:
            (lat, lon) <- inverse projection of g
            h          <- DEM(lat, lon)
            P          <- geodetic_to_ecef(lat, lon, h)
            t*         <- solve V(t).(P - S(t)) = 0        # Newton / bisection
            R          <- |P - S(t*)|
            (line,pix) <- image coordinates from (t*, R)
            output[g]  <- bilinear(image, line, pix)       # in LINEAR power

    Forward projection -- pushing each input pixel to where it lands -- leaves holes
    where ground-range sampling stretched and piles multiple inputs where it compressed.
    Iterating over the output guarantees each cell is filled exactly once by exactly one
    interpolation. Same argument as inverse warping in image processing.

    **Geocoding is not radiometric terrain correction.** Geocoding moves pixels; it does
    not change their values. RTC additionally fixes brightness: a slope facing the sensor
    is foreshortened, so each pixel collects backscatter from more ground and is brighter
    -- not because the surface scatters more. The modern area-based correction is

    .. math:: \gamma^0_T = \sigma^0_E \cdot \frac{A_{\text{flat}}}{A_{\text{illuminated}}}

    which handles several terrain facets contributing to one pixel, as the naive
    :math:`1/\cos\theta_{\mathrm{loc}}` cannot.

    **And the result is still untrustworthy in layover.** Where
    :math:`\theta_{\mathrm{loc}} \le 0`, returns from multiple distinct ground patches are
    summed in a single range bin. No division by an area recovers information destroyed
    in the receiver. Layover is not a radiometric error to correct; it is destroyed
    information to mask. That is ``P1-M1``'s comprehension question.

    TODO(student)
    -------------
    1. Solve the zero-Doppler condition by Newton's method on the orbit state vectors.
    2. Backward-geocode onto the analysis grid, interpolating in **linear power**.
    3. Apply area-based RTC using the DEM-derived illuminated area.
    4. Validate by SAR simulation cross-correlation (``docs/05`` section 5.1): the peak
       must lie at (0, 0) to within a fraction of a pixel.
    """
    raise NotImplementedError("Implement terrain correction. See docs/04 section 5.")
