r"""The physics baseline: thresholding, change detection and the decision rule.

The baseline exists to answer one question honestly: **what did the machine learning
actually buy?** A fusion network scoring 0.71 IoU is impressive until a well-tuned
threshold scores 0.68, at which point the contribution is 0.03 IoU for 25 GPU-hours and
19 M parameters.

Row 2 of ``MILESTONES.md`` section 6 is therefore approximately what operational systems
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

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TileThresholdResult",
    "ashman_d",
    "baseline_flood_map",
    "otsu_threshold",
    "tile_adaptive_threshold",
]


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    r"""Otsu's threshold: maximise between-class variance.

    .. math::

        \sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\bigl[\mu_1(t) - \mu_0(t)\bigr]^2,
        \qquad t^\star = \arg\max_t \sigma_B^2(t)

    Notes
    -----
    **Implement in** :math:`O(\mathrm{bins})` **using cumulative sums**, not the
    :math:`O(\mathrm{bins}^2)` double loop. ``docs/06`` exercise 4 asks you to verify
    the fast version against a brute-force one.

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

    TODO(student)
    -------------
    1. Histogram, then cumulative sums of counts and of ``count * value``.
    2. Evaluate :math:`\sigma_B^2` at every bin edge in one vectorised pass.
    3. Return the maximising threshold in the units of ``values`` (dB), not a bin index.
    """
    raise NotImplementedError("Implement Otsu. See docs/06 section 3.2.")


def ashman_d(mu1: float, mu2: float, sigma1: float, sigma2: float) -> float:
    r"""Ashman's :math:`D`, a separation statistic for a two-component mixture.

    .. math:: D = \sqrt{2}\,\frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}

    :math:`D > 2` conventionally indicates two distinguishable populations. Used by
    :func:`tile_adaptive_threshold` to decide whether a tile is genuinely bimodal.

    TODO(student)
    -------------
    Implement, and guard against a zero denominator from a degenerate fit.
    """
    raise NotImplementedError("Implement Ashman's D. See docs/06 section 4.2.")


@dataclass(frozen=True)
class TileThresholdResult:
    """Per-tile threshold and the acceptance evidence."""

    thresholds: np.ndarray  # (n_tiles_y, n_tiles_x), NaN where rejected
    accepted: np.ndarray  # (n_tiles_y, n_tiles_x) bool
    threshold_surface: np.ndarray  # (H, W), interpolated
    accepted_fraction: float


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

    TODO(student)
    -------------
    1. Partition into tiles; skip tiles below ``min_valid_fraction``.
    2. Fit a 2-component Gaussian mixture per tile.
    3. Apply conditions (a)--(c). ``docs/06`` exercise 5 asks you to report the IoU
       change from disabling (b) — that is the measurement that justifies this rule.
    4. Otsu (or the GMM crossing point) inside each accepted tile.
    5. Interpolate over tile centres; extrapolate conservatively at the edges.
    6. Report ``accepted_fraction`` and plot the surface.
    """
    raise NotImplementedError("Implement tile-adaptive thresholding. See docs/06 section 4.")


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

    TODO(student)
    -------------
    1. Implement the rule exactly as written, in that order.
    2. Handle ``mndwi is None`` (no optical for this chip) by falling through to
       ``water_sar`` — this is the graceful-degradation path that ablation row 11 tests
       in the learned model, and the baseline should have it too.
    3. Produce the per-mask attribution table of ``docs/06`` section 6 by re-running with
       each exclusion disabled in turn.
    """
    raise NotImplementedError("Implement the decision rule. See docs/06 section 5.1.")
