r"""The physics baseline: Otsu, tile acceptance and the decision rule.

The specification for :mod:`floodfusion.baselines`. Milestone ``P1-M2``.

The baseline exists to answer one question honestly: **what did the machine learning
actually buy?** Row 2 is approximately what operational systems do, and a well-built
version is hard to beat. Target IoU >= 0.55; if you score far above 0.75, your labels
may be threshold-derived (``docs/05`` section 3.4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.baselines import (
    ashman_d,
    baseline_flood_map,
    otsu_threshold,
    tile_adaptive_threshold,
)
from floodfusion.config import BaselineConfig

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_ashman_threshold_convention",
    "test_change_threshold_follows_from_speckle_statistics",
    "test_row1_disables_every_exclusion",
    "test_delta_min_must_be_positive",
    "test_component_weight_bounds",
    "test_unknown_threshold_mode_rejected",
})


class TestOtsu:
    def test_separates_a_clean_bimodal_histogram(self) -> None:
        rng = np.random.default_rng(0)
        water = rng.normal(-20.0, 1.0, 5000)
        land = rng.normal(-8.0, 1.5, 5000)
        t = otsu_threshold(np.concatenate([water, land]))
        assert -20.0 < t < -8.0

    def test_fails_on_a_realistic_two_percent_water_scene(self) -> None:
        """**Ablation row 1.** The water mode is a bump on the tail, not a peak.

        Otsu maximises a between-class variance dominated by the land distribution and
        places the threshold far too high. This is a result to REPORT, not a bug to fix
        (``docs/06`` section 3.3).
        """
        rng = np.random.default_rng(1)
        water = rng.normal(-20.0, 1.0, 200)     # 2 %
        land = rng.normal(-8.0, 2.5, 9800)
        t = otsu_threshold(np.concatenate([water, land]))
        assert t > -16.0, "expected an over-permissive threshold on an imbalanced scene"

    def test_returns_a_value_in_input_units(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(-12.0, 3.0, 4000)
        t = otsu_threshold(values)
        assert values.min() <= t <= values.max(), "return dB, not a bin index"


class TestAshmanD:
    def test_ashman_threshold_convention(self) -> None:
        """D > 2 indicates two distinguishable populations."""
        well_separated = math.sqrt(2) * abs(-20.0 - (-8.0)) / math.sqrt(1.0**2 + 1.5**2)
        assert well_separated > 2.0
        overlapping = math.sqrt(2) * abs(-12.0 - (-11.5)) / math.sqrt(2.0**2 + 2.0**2)
        assert overlapping < 2.0

    def test_matches_the_analytic_value(self) -> None:
        assert ashman_d(-20.0, -8.0, 1.0, 1.5) == pytest.approx(
            math.sqrt(2) * 12.0 / math.sqrt(1.0 + 2.25), rel=1e-6
        )

    def test_identical_components_give_zero(self) -> None:
        assert ashman_d(-12.0, -12.0, 2.0, 2.0) == pytest.approx(0.0)


class TestTileAcceptance:
    def test_rejects_a_pure_land_tile(self) -> None:
        """**The step everyone omits.** Most tiles contain no water at all."""
        rng = np.random.default_rng(3)
        land_only = rng.normal(-8.0, 1.5, (200, 200))
        valid = np.ones_like(land_only, dtype=bool)
        result = tile_adaptive_threshold(land_only, valid, tile_size=200)
        assert result.accepted_fraction == pytest.approx(0.0), (
            "a pure-land tile is not bimodal and must be rejected, not thresholded"
        )

    def test_accepts_a_genuinely_bimodal_tile(self) -> None:
        rng = np.random.default_rng(4)
        tile = rng.normal(-8.0, 1.5, (200, 200))
        tile[:80, :] = rng.normal(-20.0, 1.0, (80, 200))   # 40 % water
        valid = np.ones_like(tile, dtype=bool)
        assert tile_adaptive_threshold(tile, valid, tile_size=200).accepted_fraction > 0.0

    def test_rejects_a_bimodal_tile_whose_dark_mode_is_not_water(self) -> None:
        """Condition (c): forest against bare field is bimodal but irrelevant."""
        rng = np.random.default_rng(5)
        tile = rng.normal(-6.0, 0.8, (200, 200))
        tile[:80, :] = rng.normal(-10.0, 0.8, (80, 200))   # dark mode still too bright
        valid = np.ones_like(tile, dtype=bool)
        result = tile_adaptive_threshold(tile, valid, tile_size=200, water_mode_max_db=-14.0)
        assert result.accepted_fraction == pytest.approx(0.0)

    def test_threshold_surface_covers_the_scene(self) -> None:
        rng = np.random.default_rng(6)
        scene = rng.normal(-8.0, 1.5, (400, 400))
        scene[:150, :] = rng.normal(-20.0, 1.0, (150, 400))
        valid = np.ones_like(scene, dtype=bool)
        result = tile_adaptive_threshold(scene, valid, tile_size=200)
        assert result.threshold_surface.shape == scene.shape


class TestDecisionRule:
    def test_change_threshold_follows_from_speckle_statistics(self) -> None:
        r"""delta_min = k*sigma_delta(L), NOT a round number (``docs/01`` section 10.2).

        At L = 4.4: sigma_dB = 4.343*sqrt(psi'(4.4)) = 2.17, and the difference of two
        independent acquisitions has sigma = sqrt(2)*2.17 = 3.07 dB. A 2-sigma test
        therefore needs ~6.1 dB.
        """
        trigamma_4_4 = 0.2500
        sigma_db = 4.343 * math.sqrt(trigamma_4_4)
        sigma_delta = math.sqrt(2.0) * sigma_db
        assert sigma_db == pytest.approx(2.17, abs=0.05)
        assert sigma_delta == pytest.approx(3.07, abs=0.05)
        assert 2 * sigma_delta == pytest.approx(BaselineConfig().delta_min_db, abs=0.3)

    def test_and_is_stricter_than_or(self) -> None:
        """Amplitude AND change: 'dark now' and 'not dark before' (``docs/06`` 5.2)."""
        post = np.array([[-20.0, -20.0], [-8.0, -8.0]])
        pre = np.array([[-8.0, -20.0], [-8.0, -20.0]])
        surface = np.full_like(post, -16.0)
        zeros = np.zeros_like(post, dtype=bool)
        out = baseline_flood_map(
            post, pre, surface, None, None,
            np.zeros_like(post), np.zeros_like(post), zeros, zeros, zeros,
            delta_min_db=6.0,
        )
        assert out[0, 0], "darkened AND dark now => flood"
        assert not out[0, 1], "dark now but ALSO dark before => permanently dark surface"

    def test_exclusions_are_applied_last(self) -> None:
        """Hard constraints, not evidence: applying earlier would be circular."""
        post = np.full((2, 2), -20.0)
        pre = np.full((2, 2), -8.0)
        surface = np.full((2, 2), -16.0)
        zeros = np.zeros((2, 2), dtype=bool)
        shadow = np.zeros((2, 2), dtype=bool); shadow[0, 0] = True
        out = baseline_flood_map(
            post, pre, surface, None, None,
            np.zeros((2, 2)), np.zeros((2, 2)), zeros, shadow, zeros,
            delta_min_db=6.0,
        )
        assert not out[0, 0], "a shadow pixel must never be reported as water"

    def test_falls_back_to_sar_when_optical_is_absent(self) -> None:
        """Graceful degradation — the baseline must have it too."""
        post = np.full((2, 2), -20.0)
        pre = np.full((2, 2), -8.0)
        surface = np.full((2, 2), -16.0)
        zeros = np.zeros((2, 2), dtype=bool)
        out = baseline_flood_map(
            post, pre, surface, None, None,
            np.zeros((2, 2)), np.zeros((2, 2)), zeros, zeros, zeros,
            delta_min_db=6.0,
        )
        assert out.all()


class TestBaselineConfigGuards:
    def test_row1_disables_every_exclusion(self) -> None:
        cfg = BaselineConfig(
            threshold_mode="global_otsu", use_change_term=False,
            use_optical_override=False, apply_terrain_masks=False,
            apply_permanent_water_mask=False,
        )
        assert cfg.threshold_mode == "global_otsu"
        assert not cfg.apply_terrain_masks

    def test_delta_min_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            BaselineConfig(delta_min_db=-6.0)

    def test_component_weight_bounds(self) -> None:
        with pytest.raises(ValueError):
            BaselineConfig(min_component_weight=0.8)

    def test_unknown_threshold_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold_mode"):
            BaselineConfig(threshold_mode="kmeans")
