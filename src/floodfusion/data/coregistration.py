r"""Co-registration measurement: ``P1-M1``'s acceptance criterion, made computable.

"It looks aligned" is not a measurement. ``P1-M1`` requires a **median absolute
displacement below one pixel** between the SAR and optical grids, measured over many
windows (``docs/05`` section 5.2) — a single global estimate hides the spatially varying
misregistration that a DEM-datum error produces.

Methods
-------
- :func:`phase_correlation` — normalised cross-power spectrum
  :math:`R = F_1\overline{F_2}/|F_1\overline{F_2}|`, whose inverse transform peaks at the
  translation; refined to sub-pixel accuracy with the **upsampled-DFT** method of
  Guizar-Sicairos et al. (2008): the cross-correlation is evaluated by matrix-multiply DFT
  on a small, finely sampled neighbourhood of the coarse peak.
- :func:`structure_image` — SAR and optical share no radiometric relationship, so
  correlate **structure** (gradient magnitude of smoothed log-intensity), not intensity.
- :func:`median_displacement` — many windows, weak-texture windows skipped, median.
- :func:`simulate_backscatter` / :func:`validate_geocoding` — the SAR-simulation check of
  ``docs/05`` section 5.1: a :math:`\cos^n\theta_{\mathrm{loc}}` image from the DEM shares
  only terrain-induced structure with the real gamma0, which is exactly what a geocoding
  check should test; the correlation peak must sit at (0, 0).

Sign convention: shifts are the displacement of ``moving`` relative to ``reference`` in
``(rows, cols)``: ``moving = np.roll(reference, (3, -2))`` gives ``(3.0, -2.0)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "median_displacement",
    "phase_correlation",
    "simulate_backscatter",
    "structure_image",
    "validate_geocoding",
]


def _upsampled_dft(data: np.ndarray, region: int, factor: int, offsets: np.ndarray) -> np.ndarray:
    """Inverse DFT of ``data`` on a ``region x region`` patch, upsampled by ``factor``."""
    out = data
    for n_items, offset in zip(data.shape[::-1], offsets[::-1]):
        kernel = (np.arange(region) - offset)[:, None] * np.fft.fftfreq(n_items, factor)
        out = np.tensordot(np.exp(2j * np.pi * kernel), out, axes=(1, -1))
    return out


def phase_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int = 20,
    window: bool = True,
) -> tuple[float, float, float]:
    """Sub-pixel translation of ``moving`` relative to ``reference``.

    Returns
    -------
    dy, dx, peak
        Displacement in pixels (rows, cols) and the normalised correlation peak height in
        (0, 1] — a low peak means there was no shared structure to lock onto.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(moving, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("reference and moving must be 2-D arrays of one shape")
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    if window:
        taper = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        a, b = a * taper, b * taper
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    product = fb * np.conj(fa)
    product /= np.maximum(np.abs(product), 1e-12)
    cross = np.fft.ifft2(product)
    peak_index = np.unravel_index(np.argmax(np.abs(cross)), cross.shape)
    shape = np.array(cross.shape)
    shifts = np.array(peak_index, dtype=np.float64)
    shifts[shifts > shape // 2] -= shape[shifts > shape // 2]
    peak = float(np.abs(cross[peak_index]))
    if upsample_factor > 1:
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        region = int(np.ceil(upsample_factor * 1.5))
        centre = np.trunc(region / 2.0)
        offsets = centre - shifts * upsample_factor
        fine = _upsampled_dft(product, region, upsample_factor, offsets)
        fine_peak = np.unravel_index(np.argmax(np.abs(fine)), fine.shape)
        shifts = shifts + (np.array(fine_peak, dtype=np.float64) - centre) / upsample_factor
        peak = float(np.abs(fine[fine_peak]) / product.size)
    return float(shifts[0]), float(shifts[1]), peak


def structure_image(image: np.ndarray, *, sigma: float = 1.5, log: bool = True) -> np.ndarray:
    """Gradient magnitude of the smoothed (log-)image: modality-independent structure."""
    from scipy import ndimage

    x = np.asarray(image, dtype=np.float64)
    if log:
        x = np.log(np.maximum(np.nan_to_num(x, nan=np.nanmedian(x)), 1e-6)) if np.nanmin(x) > 0 else x
    x = np.nan_to_num(x, nan=float(np.nanmean(x)))
    x = ndimage.gaussian_filter(x, sigma)
    return np.hypot(ndimage.sobel(x, axis=0), ndimage.sobel(x, axis=1))


def median_displacement(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    window: int = 128,
    step: int = 64,
    min_peak: float = 0.05,
    min_texture: float = 1e-6,
    upsample_factor: int = 20,
) -> dict[str, Any]:
    """Median absolute displacement over many windows (``P1-M1``: must be < 1 px).

    Windows with no texture, or whose correlation peak is below ``min_peak`` (nothing
    shared to register), are skipped and counted. Pass structure images (see
    :func:`structure_image`) for cross-modal pairs.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(moving, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("reference and moving must share one shape")
    shifts = []
    skipped = 0
    for r in range(0, a.shape[0] - window + 1, step):
        for c in range(0, a.shape[1] - window + 1, step):
            wa, wb = a[r:r + window, c:c + window], b[r:r + window, c:c + window]
            if np.nanstd(wa) < min_texture or np.nanstd(wb) < min_texture:
                skipped += 1
                continue
            dy, dx, peak = phase_correlation(wa, wb, upsample_factor=upsample_factor)
            if peak < min_peak:
                skipped += 1
                continue
            shifts.append((dy, dx))
    if not shifts:
        return {"n_windows": 0, "skipped": skipped, "median_abs_displacement": float("nan"),
                "median_dy": float("nan"), "median_dx": float("nan"), "shifts": np.empty((0, 2))}
    s = np.asarray(shifts)
    return {
        "n_windows": len(shifts),
        "skipped": skipped,
        "median_dy": float(np.median(s[:, 0])),
        "median_dx": float(np.median(s[:, 1])),
        "median_abs_displacement": float(np.median(np.hypot(s[:, 0], s[:, 1]))),
        "shifts": s,
    }


def simulate_backscatter(
    theta_loc_deg: np.ndarray,
    *,
    exponent: float = 1.5,
    shadow: np.ndarray | None = None,
    layover: np.ndarray | None = None,
) -> np.ndarray:
    r"""Simulated relative backscatter :math:`\cos^n\theta_{\mathrm{loc}}` in dB.

    Absolute values do not matter, structure does (``docs/05`` section 5.1). Shadow is set
    to the darkest value and layover to the brightest, as a radar image shows them.
    """
    cos = np.cos(np.radians(np.asarray(theta_loc_deg, dtype=np.float64)))
    db = 10.0 * np.log10(np.clip(cos, 1e-3, 1.0) ** exponent)
    if shadow is not None:
        db = np.where(shadow, np.nanmin(db), db)
    if layover is not None:
        db = np.where(layover, np.nanmax(db), db)
    return db


def validate_geocoding(
    gamma0_db: np.ndarray,
    theta_loc_deg: np.ndarray,
    *,
    shadow: np.ndarray | None = None,
    layover: np.ndarray | None = None,
    window: int = 256,
    step: int = 128,
) -> dict[str, Any]:
    """Cross-correlate a terrain simulation against real gamma0: the peak must be at (0, 0).

    A systematic offset means the terrain correction is misregistered — usually a DEM
    vertical-datum error (50 m of geoid/ellipsoid confusion is ~62 m, six pixels, at 39
    degrees) or the wrong orbit file (``docs/05`` section 5.1).
    """
    simulated = simulate_backscatter(theta_loc_deg, shadow=shadow, layover=layover)
    return median_displacement(
        structure_image(simulated, log=False), structure_image(gamma0_db, log=False),
        window=window, step=step,
    )
