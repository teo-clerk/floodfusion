r"""The physics baseline: thresholding, change detection and the decision rule.

The baseline exists to answer one question honestly: **what did the machine learning
actually buy?** A fusion network scoring 0.71 IoU is impressive until a well-tuned
threshold scores 0.68, at which point the contribution is 0.03 IoU for 25 GPU-hours and
19 M parameters.

Ablation row 2 is therefore approximately what operational systems
do — tile-adaptive thresholding plus change detection plus the terrain and
permanent-water masks. It is genuinely hard to beat, and that is the point. Target:
IoU :math:`\ge 0.55`. If you land far above 0.75, apply the label-provenance tests of
``docs/05`` section 3.4 before celebrating.

References
----------
``docs/06_baselines_and_metrics.md`` sections 3, 4, 5, 6.
``docs/01_sar_hydrology_and_water_backscatter.md`` sections 10, 11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GaussianMixture1D",
    "TileThresholdResult",
    "ashman_d",
    "baseline_flood_map",
    "fit_gmm_1d",
    "otsu_threshold",
    "region_grow",
    "tile_adaptive_threshold",
]


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    r"""Otsu's threshold: maximise between-class variance.

    .. math::

        \sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\bigl[\mu_1(t) - \mu_0(t)\bigr]^2,
        \qquad t^\star = \arg\max_t \sigma_B^2(t)

    Notes
    -----
    **Implemented in** :math:`O(\mathrm{bins})` **with cumulative sums**, not the
    :math:`O(\mathrm{bins}^2)` double loop; ``tests/test_physics_extra.py`` checks it
    against a brute-force version.

    **Expect it to fail on a full scene, and report that as a result.** Its three
    assumptions all break here (``docs/01`` section 11.2):

    1. *The histogram is bimodal.* In a scene that is 2 % water, the water mode is a
       bump on the tail of the land mode, not a peak.
    2. *The classes have comparable variance.* Water is tightly distributed; land is
       broad and multi-modal. Otsu is biased toward the wider class.
    3. *One threshold suits the whole image.* Incidence angle varies across the swath
       and land cover varies spatially.

    The observed consequence is a threshold several dB too high and a badly over-detected
    map. That is **ablation row 1** — a number to record, not a bug to hide.

    Implementation notes
    --------------------
    1. Histogram, then cumulative sums of counts and of ``count * value``.
    2. :math:`\sigma_B^2` is evaluated at every bin edge in one vectorised pass.
    3. The maximising threshold is returned in the units of ``values`` (dB), not as a bin index.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("otsu_threshold needs at least two finite values")
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return lo
    counts, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = counts.astype(np.float64)
    w0 = np.cumsum(counts)                  # pixels at or below each bin
    w1 = w0[-1] - w0
    m0 = np.cumsum(counts * centres)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu1 - mu0) ** 2    # proportional to sigma_B^2; one vectorised pass
    k = int(np.argmax(between[:-1]))        # threshold sits on the upper edge of bin k
    return float(edges[k + 1])


def ashman_d(mu1: float, mu2: float, sigma1: float, sigma2: float) -> float:
    r"""Ashman's :math:`D`, a separation statistic for a two-component mixture.

    .. math:: D = \sqrt{2}\,\frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}

    :math:`D > 2` conventionally indicates two distinguishable populations. Used by
    :func:`tile_adaptive_threshold` to decide whether a tile is genuinely bimodal.

    Implementation notes
    --------------------
    A degenerate fit (zero denominator) gives 0 for identical means and ``inf`` otherwise.
    """
    denom = math.sqrt(sigma1**2 + sigma2**2)
    if denom == 0.0:
        return 0.0 if mu1 == mu2 else math.inf
    return math.sqrt(2.0) * abs(mu1 - mu2) / denom


@dataclass(frozen=True)
class GaussianMixture1D:
    """Two-component 1-D Gaussian mixture, component 0 the DARKER (lower-mean) one."""

    weights: tuple[float, float]
    means: tuple[float, float]
    sigmas: tuple[float, float]
    converged: bool
    iterations: int

    @property
    def ashman_d(self) -> float:
        return ashman_d(self.means[0], self.means[1], self.sigmas[0], self.sigmas[1])


#: Above this many samples the EM runs on a histogram (see :func:`fit_gmm_1d`).
GMM_EXACT_MAX_SAMPLES = 4096
GMM_HISTOGRAM_BINS = 1024


def _em_two_gaussians(
    x: np.ndarray, n: np.ndarray, w: np.ndarray, mu: np.ndarray, var: np.ndarray, *,
    max_iter: int, tol: float, min_sigma: float, var_correction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """Weighted EM: ``x`` sample values (or bin centres), ``n`` their multiplicities."""
    total = float(n.sum())
    previous, converged, it = -np.inf, False, 0
    for it in range(1, max_iter + 1):  # noqa: B007 - `it` is reported after the loop
        log_pdf = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2.0 * np.pi * var))
        log_joint = log_pdf + np.log(np.maximum(w, 1e-300))
        log_norm = np.logaddexp(log_joint[:, 0], log_joint[:, 1])
        resp = np.exp(log_joint - log_norm[:, None]) * n[:, None]
        nk = resp.sum(axis=0) + 1e-12
        w = nk / total
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = np.maximum((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk + var_correction, min_sigma**2)
        ll = float((n * log_norm).sum() / total)
        if abs(ll - previous) <= tol * max(1.0, abs(ll)):
            converged = True
            break
        previous = ll
    return w, mu, var, converged, it


def fit_gmm_1d(
    values: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    min_sigma: float = 1e-3,
) -> GaussianMixture1D:
    """Deterministic EM for a two-component 1-D Gaussian mixture.

    Initialised from the Otsu split (so the result is reproducible — no random restarts),
    iterated to a relative log-likelihood change below ``tol``. Variances are floored at
    ``min_sigma**2`` so a degenerate component cannot collapse onto one value.

    Above :data:`GMM_EXACT_MAX_SAMPLES` values the E and M steps run on a
    :data:`GMM_HISTOGRAM_BINS`-bin histogram with the counts as weights, plus Sheppard's
    correction :math:`h^2/12` to the variances: identical to the per-sample EM up to the
    bin width (~0.03 dB over a 30 dB range — far below the speckle), and independent of the
    tile size. A 200 x 200 tile then costs 1024 instead of 40 000 evaluations per iteration.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        raise ValueError("fit_gmm_1d needs at least four finite values")
    t = otsu_threshold(x)
    lo, hi = x[x <= t], x[x > t]
    if lo.size == 0 or hi.size == 0:
        lo, hi = np.sort(x)[: x.size // 2], np.sort(x)[x.size // 2:]
    w = np.array([lo.size, hi.size], dtype=np.float64) / x.size
    mu = np.array([lo.mean(), hi.mean()])
    var = np.maximum(np.array([lo.var(), hi.var()]), min_sigma**2)
    if x.size > GMM_EXACT_MAX_SAMPLES and x.max() > x.min():
        counts, edges = np.histogram(x, bins=GMM_HISTOGRAM_BINS)
        keep = counts > 0
        centres = 0.5 * (edges[:-1] + edges[1:])[keep]
        width = float(edges[1] - edges[0])
        w, mu, var, converged, it = _em_two_gaussians(centres, counts[keep].astype(np.float64), w, mu, var,
                                                      max_iter=max_iter, tol=tol, min_sigma=min_sigma,
                                                      var_correction=width**2 / 12.0)
    else:
        w, mu, var, converged, it = _em_two_gaussians(x, np.ones_like(x), w, mu, var, max_iter=max_iter,
                                                      tol=tol, min_sigma=min_sigma, var_correction=0.0)
    order = np.argsort(mu)
    return GaussianMixture1D(
        weights=(float(w[order[0]]), float(w[order[1]])),
        means=(float(mu[order[0]]), float(mu[order[1]])),
        sigmas=(float(np.sqrt(var[order[0]])), float(np.sqrt(var[order[1]]))),
        converged=converged,
        iterations=it,
    )


@dataclass(frozen=True)
class TileThresholdResult:
    """Per-tile threshold and the acceptance evidence."""

    thresholds: np.ndarray  # (n_tiles_y, n_tiles_x), NaN where rejected
    accepted: np.ndarray  # (n_tiles_y, n_tiles_x) bool
    threshold_surface: np.ndarray  # (H, W), interpolated
    accepted_fraction: float
    #: Per-tile acceptance evidence, NaN where a tile was skipped for low valid fraction:
    #: ``ashman_d``, ``dark_weight``, ``dark_mean_db`` — what you plot to justify the rule.
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


def tile_adaptive_threshold(
    gamma0_db: np.ndarray,
    valid: np.ndarray,
    *,
    tile_size: int = 200,
    min_valid_fraction: float = 0.5,
    ashman_min: float = 2.0,
    min_component_weight: float = 0.10,
    water_mode_max_db: float = -14.0,
) -> TileThresholdResult:
    r"""Spatially varying threshold with an explicit tile-acceptance rule.

    Notes
    -----
    Tiling helps for two independent reasons already established:

    1. **Incidence angle varies across the swath** (29--46 degrees), so the physically
       correct threshold varies across the image even for identical surfaces.
    2. **Land cover varies spatially**, so the *land* mode that corrupts global Otsu
       differs between an urban tile and an agricultural one.

    .. warning::

       **The tile-acceptance rule is the step everyone omits, and the one that makes the
       difference.** Most tiles contain no water at all; running Otsu on them yields a
       meaningless threshold that splits the land distribution in half. Accepting those
       makes naive tiling no better than global Otsu.

    Accept a tile only if **all** of:

    a. Ashman's :math:`D > 2` — the two fitted components are separable;
    b. :math:`\min(w_1, w_2) > 0.10` and :math:`\max(w_1,w_2) < 0.90` — the water
       fraction is plausible, neither component negligible;
    c. :math:`\min(\mu_1, \mu_2) <` ``water_mode_max_db`` — the dark mode is actually
       dark enough to be water.

    Condition (c) guards against a tile that is bimodal for an irrelevant reason, such as
    forest against bare field. Without it you threshold land against land and the
    interpolated surface is corrupted by tiles that should have been rejected.

    Then **interpolate** the accepted thresholds into a smooth surface, apply it, and
    **region-grow** from high-confidence seeds to recover boundary pixels lost to speckle.

    **Filter first.** At :math:`L = 4.4`, :math:`\sigma_{\mathrm{dB}} = 2.17` dB, so a
    single-pixel threshold is dominated by noise (``docs/01`` section 10.2). Use Refined
    Lee and record the window size — it changes the effective :math:`L` and therefore the
    result.

    Implementation notes
    --------------------
    1. Partition into tiles; tiles below ``min_valid_fraction`` are skipped.
    2. A 2-component Gaussian mixture is fitted per tile.
    3. Conditions (a)--(c) decide whether the tile is accepted as bimodal.
    4. Otsu inside each accepted tile.
    5. Thresholds are interpolated over the tile centres by inverse-distance weighting
       (:func:`_idw_surface`).
    6. ``accepted_fraction`` is returned with the surface.
    """
    db = np.asarray(gamma0_db, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(db)
    if db.shape != ok.shape or db.ndim != 2:
        raise ValueError("gamma0_db and valid must be 2-D arrays of one shape")
    if tile_size < 8:
        raise ValueError("tile_size must be at least 8 pixels")
    h, w = db.shape
    ny, nx = math.ceil(h / tile_size), math.ceil(w / tile_size)
    thresholds = np.full((ny, nx), np.nan)
    accepted = np.zeros((ny, nx), dtype=bool)
    diag = {k: np.full((ny, nx), np.nan) for k in ("ashman_d", "dark_weight", "dark_mean_db")}
    for iy in range(ny):
        for ix in range(nx):
            sl = (slice(iy * tile_size, (iy + 1) * tile_size), slice(ix * tile_size, (ix + 1) * tile_size))
            tile_ok = ok[sl]
            if tile_ok.mean() < min_valid_fraction or tile_ok.sum() < 16:
                continue
            values = db[sl][tile_ok]
            gmm = fit_gmm_1d(values)
            d = gmm.ashman_d
            dark_w = gmm.weights[0]
            diag["ashman_d"][iy, ix] = d
            diag["dark_weight"][iy, ix] = dark_w
            diag["dark_mean_db"][iy, ix] = gmm.means[0]
            accept = (
                d > ashman_min                                           # (a) separable
                and min_component_weight < dark_w < 1.0 - min_component_weight  # (b) plausible fraction
                and gmm.means[0] < water_mode_max_db                     # (c) dark mode is water-dark
            )
            if accept:
                thresholds[iy, ix] = otsu_threshold(values)
                accepted[iy, ix] = True
    surface = _idw_surface(thresholds, accepted, tile_size, (h, w))
    return TileThresholdResult(
        thresholds=thresholds,
        accepted=accepted,
        threshold_surface=surface,
        accepted_fraction=float(accepted.sum() / accepted.size),
        diagnostics=diag,
    )


def _idw_surface(
    thresholds: np.ndarray, accepted: np.ndarray, tile_size: int, shape: tuple[int, int],
    *, power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance-weighted threshold surface over accepted tile centres.

    Evaluated on a coarse lattice (every ``tile_size // 8`` pixels) and bilinearly
    upsampled — cheap for a full IW scene, and smooth. IDW extrapolates conservatively:
    beyond the accepted tiles it tends to the nearest accepted values rather than
    trending. Returns all-NaN if no tile was accepted ("no bimodal evidence anywhere").
    """
    from floodfusion.data.preprocessing import interpolate_lut

    h, w = shape
    if not accepted.any():
        return np.full(shape, np.nan)
    iy, ix = np.nonzero(accepted)
    cy = (iy + 0.5) * tile_size
    cx = (ix + 0.5) * tile_size
    values = thresholds[iy, ix]
    step = max(1, tile_size // 8)
    rows = np.unique(np.append(np.arange(0, h, step), h - 1)).astype(np.float64)
    cols = np.unique(np.append(np.arange(0, w, step), w - 1)).astype(np.float64)
    dist2 = (rows[:, None, None] - cy) ** 2 + (cols[None, :, None] - cx) ** 2
    weights = 1.0 / np.maximum(dist2, 1.0) ** (power / 2.0)
    coarse = (weights * values).sum(axis=-1) / weights.sum(axis=-1)
    return interpolate_lut(coarse, shape, lut_lines=rows, lut_pixels=cols)


def region_grow(seeds: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Keep every 8-connected component of ``candidates`` that contains a seed, plus seeds.

    The high-confidence seeds come from the strict decision; the candidates are pixels
    moderately past the threshold. Growing recovers boundary pixels lost to speckle
    without admitting isolated dark blobs that never touch confident water
    (``docs/06`` section 4.2, step 7).
    """
    from scipy import ndimage

    seeds = np.asarray(seeds, dtype=bool)
    candidates = np.asarray(candidates, dtype=bool) | seeds
    labels, n = ndimage.label(candidates, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return seeds.copy()
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def baseline_flood_map(
    gamma0_vv_post_db: np.ndarray,
    gamma0_vv_pre_db: np.ndarray,
    threshold_surface: np.ndarray,
    mndwi: np.ndarray | None,
    optical_valid: np.ndarray | None,
    slope_deg: np.ndarray,
    hand_m: np.ndarray,
    layover: np.ndarray,
    shadow: np.ndarray,
    permanent_water: np.ndarray,
    *,
    delta_min_db: float,
    tau_mndwi: float = 0.0,
    slope_max_deg: float = 10.0,
    hand_max_m: float = 15.0,
    use_change_term: bool = True,
    combine: str = "and",
    grow_margin_db: float | None = None,
    apply_terrain_masks: bool = True,
    apply_permanent_water_mask: bool = True,
) -> np.ndarray:
    r"""The full baseline decision rule.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` boolean flood mask.

    Notes
    -----
    The rule (``docs/06`` section 5.1)::

        water_amp    <- gamma0_VV_post_db < threshold_surface
        water_change <- (post - pre) < -delta_min_db
        water_sar    <- water_amp AND water_change

        optical_ok   <- NOT (cloud OR cloud_shadow)
        water_opt    <- MNDWI > tau_mndwi

        water        <- WHERE(optical_ok, water_opt, water_sar)

        water <- water AND (slope < slope_max) AND (hand < hand_max)
        water <- water AND NOT layover AND NOT shadow AND NOT permanent_water

    **Three design choices you must be able to defend:**

    *Why AND and not OR in the SAR decision.* Amplitude alone accepts every permanently
    dark surface — tarmac, dry sand, playa (``docs/01`` section 9.2). Change alone
    accepts every darkening, including a harvested field. Requiring both is the
    conjunction of "it is dark now" and "it was not dark before". The cost is recall:
    flooding of an already-dark surface is missed. Given the cost asymmetry of
    ``docs/00`` section 6, **measure** the OR variant and report both.

    *Why optical overrides rather than votes.* A valid optical observation is nearly
    unambiguous while SAR is a proxy (``docs/02`` section 1). A weighted vote would
    dilute a near-certain observation with an ambiguous one. Note that the learned model
    does something more subtle — it learns a *soft, per-pixel* version of exactly this
    override (``docs/07`` section 8).

    *Why exclusions are applied last.* They are hard physical constraints, not evidence.
    Applying them earlier would let them influence threshold estimation, which is
    circular.

    ``delta_min_db`` must come from the **speckle statistics**, not a round number:
    :math:`k\,\sigma_{\Delta,\mathrm{dB}}(L_{\mathrm{eff}})` with :math:`k \approx 2`--3
    (``docs/01`` section 10.2).

    Every threshold here is tuned on **validation** and frozen. Tuning on test makes the
    test set a validation set. Record the tuned values in the config: a baseline you
    cannot regenerate is not a baseline.

    Implementation notes
    --------------------
    1. The rule is applied exactly as written, in that order.
    2. ``mndwi is None`` (no optical for this chip) falls through to ``water_sar`` — the
       graceful-degradation path that ablation row 11 tests in the learned model.
    3. The per-mask attribution table of ``docs/06`` section 6 re-runs the rule with each
       exclusion disabled in turn (:mod:`floodfusion.baseline_runner`).
    """
    if delta_min_db <= 0.0:
        raise ValueError("delta_min_db is a magnitude and must be positive")
    if combine not in {"and", "or"}:
        raise ValueError(f"combine must be 'and' or 'or', got {combine!r}")
    post = np.asarray(gamma0_vv_post_db, dtype=np.float64)
    pre = np.asarray(gamma0_vv_pre_db, dtype=np.float64)
    surface = np.asarray(threshold_surface, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        # 1. SAR amplitude evidence ("dark now"). NaN surface (no accepted tile) -> False.
        water_amp = post < surface
        # 2. Bi-temporal change evidence ("not dark before"): a DROP of more than delta_min.
        change = post - pre
        water_change = change < -delta_min_db
    # 3. SAR decision. AND by default: amplitude alone admits every permanently dark
    #    surface, change alone every darkening (docs/06 section 5.2).
    if not use_change_term:
        water_sar = water_amp
    elif combine == "and":
        water_sar = water_amp & water_change
    else:
        water_sar = water_amp | water_change
    if grow_margin_db:
        with np.errstate(invalid="ignore"):
            candidates = post < surface + grow_margin_db
            if use_change_term:
                candidates &= change < -0.5 * delta_min_db  # never grow into permanently dark
        water_sar = region_grow(water_sar, candidates)
    # 4-5. Optical evidence overrides where it is valid; SAR alone elsewhere — the
    #      graceful-degradation path (mndwi is None => no optical for this scene).
    if mndwi is not None and optical_valid is not None:
        mndwi = np.asarray(mndwi, dtype=np.float64)
        optical_ok = np.asarray(optical_valid, dtype=bool) & np.isfinite(mndwi)
        with np.errstate(invalid="ignore"):
            water = np.where(optical_ok, mndwi > tau_mndwi, water_sar)
    else:
        water = water_sar.copy()
    # 6. Hard physical exclusions, LAST: they are constraints, not evidence.
    if apply_terrain_masks:
        slope = np.asarray(slope_deg, dtype=np.float64)
        hand_arr = np.asarray(hand_m, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            water &= ~(slope >= slope_max_deg)
            water &= ~(hand_arr >= hand_max_m)
        water &= ~np.asarray(layover, dtype=bool)
        water &= ~np.asarray(shadow, dtype=bool)
    if apply_permanent_water_mask:
        water &= ~np.asarray(permanent_water, dtype=bool)
    return water
