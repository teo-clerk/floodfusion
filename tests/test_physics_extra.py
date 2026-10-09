"""Behaviour beyond the given specification: LUT calibration, Refined Lee at edges,
thermal-noise accounting, the 1-D mixture fit, region growing, the AND/OR rule, HAND
along flow paths, the validity flag band and permanent water.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.baselines import (
    baseline_flood_map,
    fit_gmm_1d,
    otsu_threshold,
    region_grow,
    tile_adaptive_threshold,
)
from floodfusion.config import ValidityFlag
from floodfusion.data.preprocessing import (
    DB_NODATA,
    calibrate,
    db_to_linear,
    equivalent_number_of_looks,
    interpolate_lut,
    linear_to_db,
    refined_lee,
    remove_thermal_noise,
    thermal_noise_report,
)
from floodfusion.masks import (
    build_validity_mask,
    hand,
    permanent_water_from_occurrence,
    validity_flag_counts,
)


class TestCalibrationLut:
    def test_coarse_lut_is_bilinearly_interpolated(self) -> None:
        lut = np.array([[100.0, 200.0], [300.0, 400.0]])
        full = interpolate_lut(lut, (5, 5), lut_lines=[0, 4], lut_pixels=[0, 4])
        assert full[0, 0] == 100.0 and full[4, 4] == 400.0
        assert full[2, 2] == pytest.approx(250.0)

    def test_coarse_lut_without_axes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="lut_lines"):
            calibrate(np.ones((4, 4)), np.ones((2, 2)))

    def test_calibration_squares_against_the_lut(self) -> None:
        dn = np.full((3, 3), 1000.0)
        out = calibrate(dn, np.full((3, 3), 500.0), "sigma")
        assert np.allclose(out, 4.0)

    def test_unknown_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            calibrate(np.ones((2, 2)), np.ones((2, 2)), "kappa")


class TestDecibels:
    def test_sentinel_round_trips_to_nan(self) -> None:
        db = linear_to_db(np.array([0.0, 1e-3]))
        assert db[0] == DB_NODATA
        lin = db_to_linear(db)
        assert np.isnan(lin[0]) and lin[1] == pytest.approx(1e-3)

    def test_mean_of_window_matches_linear_domain_mean(self) -> None:
        """docs/04 3.4: average in linear power, convert last."""
        rng = np.random.default_rng(0)
        power = 0.05 * rng.gamma(4.4, 1 / 4.4, 200_000)
        correct = linear_to_db(np.array([power.mean()]))[0]
        biased = linear_to_db(power).mean()
        assert correct == pytest.approx(10 * math.log10(0.05), abs=0.03)
        assert biased - correct == pytest.approx(-0.51, abs=0.05)


class TestThermalNoise:
    def test_report_counts_noise_limited_pixels(self) -> None:
        out = remove_thermal_noise(np.array([1e-3, 5e-3, 2e-3]), np.array([2e-3, 1e-3, 2e-3]))
        report = thermal_noise_report(out)
        assert report["non_positive"] == 2 and report["total"] == 3


class TestRefinedLee:
    def test_edges_survive_filtering(self) -> None:
        rng = np.random.default_rng(3)
        clean = np.ones((64, 64))
        clean[:, 32:] = 10.0
        noisy = clean * rng.gamma(4.4, 1 / 4.4, clean.shape)
        out = refined_lee(noisy, 4.4, 7)
        assert out[:, 26:31].mean() < 1.5, "dark side must not be bled into by the bright side"
        assert out[:, 33:38].mean() > 8.5

    def test_nodata_is_left_untouched(self) -> None:
        rng = np.random.default_rng(4)
        img = rng.gamma(4.4, 1 / 4.4, (32, 32))
        img[10, 10] = np.nan
        img[5, 5] = 0.0
        out = refined_lee(img, 4.4, 7)
        assert np.isnan(out[10, 10]) and out[5, 5] == 0.0
        assert np.isfinite(out[np.isfinite(img)]).all()

    def test_enl_reports_looks(self) -> None:
        rng = np.random.default_rng(5)
        assert equivalent_number_of_looks(rng.gamma(4.4, 1 / 4.4, 100_000)) == pytest.approx(4.4, rel=0.05)

    def test_rejects_even_window(self) -> None:
        with pytest.raises(ValueError):
            refined_lee(np.ones((8, 8)), 4.4, 6)


class TestMixtureAndThresholds:
    def test_gmm_recovers_two_components(self) -> None:
        rng = np.random.default_rng(6)
        values = np.concatenate([rng.normal(-20, 1, 3000), rng.normal(-8, 1.5, 7000)])
        gmm = fit_gmm_1d(values)
        assert gmm.means[0] == pytest.approx(-20, abs=0.2)
        assert gmm.means[1] == pytest.approx(-8, abs=0.2)
        assert gmm.weights[0] == pytest.approx(0.3, abs=0.02)
        assert gmm.ashman_d > 2

    def test_otsu_on_constant_input(self) -> None:
        assert otsu_threshold(np.full(10, -12.0)) == -12.0

    def test_otsu_matches_brute_force(self) -> None:
        """docs/06 exercise 4: the O(bins) version agrees with O(bins^2)."""
        rng = np.random.default_rng(7)
        values = np.concatenate([rng.normal(-18, 2, 500), rng.normal(-9, 2, 1500)])
        counts, edges = np.histogram(values, bins=64)
        centres = 0.5 * (edges[:-1] + edges[1:])
        best, best_k = -1.0, 0
        for k in range(63):
            w0, w1 = counts[: k + 1].sum(), counts[k + 1:].sum()
            mu0 = (counts[: k + 1] * centres[: k + 1]).sum() / w0
            mu1 = (counts[k + 1:] * centres[k + 1:]).sum() / w1
            sb = w0 * w1 * (mu1 - mu0) ** 2
            if sb > best:
                best, best_k = sb, k
        assert otsu_threshold(values, n_bins=64) == pytest.approx(edges[best_k + 1])

    def test_no_accepted_tile_gives_nan_surface(self) -> None:
        rng = np.random.default_rng(8)
        land = rng.normal(-8, 1.5, (100, 100))
        res = tile_adaptive_threshold(land, np.ones_like(land, bool), tile_size=50)
        assert res.accepted_fraction == 0.0 and np.isnan(res.threshold_surface).all()

    def test_surface_interpolates_between_accepted_tiles(self) -> None:
        rng = np.random.default_rng(9)
        scene = rng.normal(-8, 1.5, (200, 400))
        scene[:150, :] = rng.normal(-20, 1, (150, 400))  # 100 % / 50 % water per tile row
        res = tile_adaptive_threshold(scene, np.ones_like(scene, bool), tile_size=100)
        assert not res.accepted[0].any(), "an all-water tile is not bimodal either"
        assert res.accepted[1].all()
        assert np.isfinite(res.threshold_surface).all()
        assert -20 < np.median(res.threshold_surface) < -8


class TestDecisionRuleVariants:
    def _scene(self):
        post = np.array([[-20.0, -20.0, -15.0]])
        pre = np.array([[-8.0, -20.0, -6.0]])
        zeros = np.zeros_like(post, dtype=bool)
        return post, pre, np.full_like(post, -16.0), zeros

    def test_or_admits_permanently_dark_surfaces(self) -> None:
        post, pre, surface, zeros = self._scene()
        kwargs = dict(delta_min_db=6.0)
        args = (post, pre, surface, None, None, np.zeros_like(post), np.zeros_like(post), zeros, zeros, zeros)
        and_map = baseline_flood_map(*args, **kwargs)
        or_map = baseline_flood_map(*args, combine="or", **kwargs)
        assert and_map.tolist() == [[True, False, False]]
        assert or_map.tolist() == [[True, True, True]]

    def test_optical_overrides_where_valid(self) -> None:
        post, pre, surface, zeros = self._scene()
        mndwi = np.array([[-0.5, 0.4, 0.4]])
        ok = np.array([[True, True, False]])
        out = baseline_flood_map(post, pre, surface, mndwi, ok, np.zeros_like(post), np.zeros_like(post),
                                 zeros, zeros, zeros, delta_min_db=6.0)
        assert out.tolist() == [[False, True, False]]

    def test_region_growing_never_enters_permanently_dark_pixels(self) -> None:
        post = np.array([[-20.0, -15.5, -15.5]])
        pre = np.array([[-8.0, -10.0, -15.5]])
        zeros = np.zeros_like(post, dtype=bool)
        out = baseline_flood_map(post, pre, np.full_like(post, -16.0), None, None, np.zeros_like(post),
                                 np.zeros_like(post), zeros, zeros, zeros, delta_min_db=6.0,
                                 grow_margin_db=1.5)
        assert out.tolist() == [[True, True, False]]

    def test_region_grow_keeps_only_seeded_components(self) -> None:
        cand = np.array([[1, 1, 0, 1, 1]], dtype=bool)
        seeds = np.array([[1, 0, 0, 0, 0]], dtype=bool)
        assert region_grow(seeds, cand).tolist() == [[True, True, False, False, False]]

    def test_masks_can_be_disabled_for_row_one(self) -> None:
        post, pre, surface, zeros = self._scene()
        shadow = np.ones_like(post, dtype=bool)
        off = baseline_flood_map(post, pre, surface, None, None, np.zeros_like(post), np.zeros_like(post),
                                 zeros, shadow, zeros, delta_min_db=6.0, use_change_term=False,
                                 apply_terrain_masks=False)
        assert off[0, 0] and off[0, 1]


class TestHandAlongFlowPath:
    def test_hand_is_zero_on_the_drainage_network(self) -> None:
        # A V-shaped valley draining north: the valley floor is the channel.
        rows, cols = np.mgrid[0:60, 0:41]
        dem = 2.0 * np.abs(cols - 20) + 0.2 * rows + 100.0
        out = hand(dem, accumulation_threshold=30, pixel_size_m=30.0)
        assert out[10:50, 20].max() < 1e-6
        # column 0 is a domain-edge outlet (drainage by convention), so compare inland cells
        assert out[30, 2] > out[30, 15] > 0.0

    def test_divide_is_respected(self) -> None:
        """Nearest along the FLOW PATH: a cell over the ridge drains to the other river."""
        rows, cols = np.mgrid[0:50, 0:61]
        ridge = 30.0 - np.abs(cols - 30)
        dem = np.where(cols < 30, ridge + 0.5 * cols, ridge + 0.1 * (60 - cols)) + 0.1 * rows
        out = hand(dem, 20, 30.0)
        assert np.isfinite(out).all() and (out >= 0).all()


class TestValidityBand:
    def test_slope_flag_can_be_omitted(self) -> None:
        zeros = np.zeros((4, 4), dtype=bool)
        steep = np.full((4, 4), 45.0)
        assert (build_validity_mask(zeros, zeros, zeros, zeros, steep, slope_max_deg=None) == 0).all()

    def test_outside_aoi_flag(self) -> None:
        zeros = np.zeros((2, 2), dtype=bool)
        out = np.array([[True, False], [False, False]])
        mask = build_validity_mask(zeros, zeros, zeros, zeros, None, slope_max_deg=None, outside_aoi=out)
        assert mask[0, 0] == ValidityFlag.OUTSIDE_AOI and mask[1, 1] == 0

    def test_counts_attribute_causes(self) -> None:
        zeros = np.zeros((2, 2), dtype=bool)
        shadow = np.array([[True, True], [False, False]])
        perm = np.array([[True, False], [False, False]])
        counts = validity_flag_counts(build_validity_mask(zeros, shadow, perm, zeros, None, slope_max_deg=None))
        assert counts["SHADOW"] == 2 and counts["PERMANENT_WATER"] == 1 and counts["INVALID"] == 2

    def test_permanent_water_from_jrc_occurrence(self) -> None:
        occ = np.array([95.0, 80.0, 79.0, 255.0])
        assert permanent_water_from_occurrence(occ).tolist() == [True, True, False, False]
