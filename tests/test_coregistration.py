"""Co-registration measurement (P1-M1: median displacement < 1 px)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from floodfusion.data.coregistration import (
    median_displacement,
    median_shift_se,
    phase_correlation,
    simulate_backscatter,
    structure_image,
    validate_geocoding,
)
from floodfusion.masks import horn_slope_aspect, local_incidence_angle


def _texture(shape=(128, 128), seed=0) -> np.ndarray:
    return ndimage.gaussian_filter(np.random.default_rng(seed).random(shape), 2.0)


def _shift(image: np.ndarray, shift) -> np.ndarray:
    return np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(image), shift)).real


class TestPhaseCorrelation:
    def test_integer_shift_and_sign_convention(self) -> None:
        base = _texture()
        dy, dx, peak = phase_correlation(base, np.roll(base, (3, -2), axis=(0, 1)), window=False)
        assert (dy, dx) == (pytest.approx(3.0), pytest.approx(-2.0)) and peak > 0.9

    @pytest.mark.parametrize("shift", [(0.5, 0.0), (-1.25, 0.4), (0.2, -0.7)])
    def test_sub_pixel_accuracy(self, shift) -> None:
        dy, dx, _ = phase_correlation(_texture(), _shift(_texture(), shift))
        assert dy == pytest.approx(shift[0], abs=0.1) and dx == pytest.approx(shift[1], abs=0.1)

    def test_unrelated_images_have_a_low_peak(self) -> None:
        _, _, peak = phase_correlation(_texture(seed=1), _texture(seed=2))
        assert peak < 0.2

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            phase_correlation(np.zeros((8, 8)), np.zeros((8, 9)))


class TestMedianDisplacement:
    def test_recovers_a_half_pixel_offset_over_many_windows(self) -> None:
        base = _texture((512, 512))
        r = median_displacement(structure_image(base, log=False), structure_image(_shift(base, (0.5, 0.0)), log=False))
        assert r["n_windows"] > 20 and r["median_abs_displacement"] == pytest.approx(0.5, abs=0.1)

    def test_textureless_windows_are_skipped(self) -> None:
        r = median_displacement(np.ones((256, 256)), np.ones((256, 256)))
        assert r["n_windows"] == 0 and r["skipped"] > 0 and np.isnan(r["median_abs_displacement"])

    def test_windows_mostly_outside_the_valid_mask_are_skipped_not_filled(self) -> None:
        # Arrange: only the right half is valid (e.g. the left half is cloud or unlabelled).
        base = _texture((256, 512))
        valid = np.zeros(base.shape, dtype=bool)
        valid[:, 256:] = True

        # Act
        r = median_displacement(structure_image(base, log=False), structure_image(_shift(base, (0.0, 1.0)), log=False),
                                valid=valid)

        # Assert: 3 rows x 3 cols of 128-px windows (step 64) lie entirely in the right half.
        assert r["n_windows"] == 9 and r["median_dx"] == pytest.approx(1.0, abs=0.1)
        assert median_displacement(np.ones((256, 256)), np.ones((256, 256)), valid=np.zeros((256, 256), bool))[
            "n_windows"] == 0

    def test_valid_mask_must_match_the_images(self) -> None:
        with pytest.raises(ValueError, match="valid"):
            median_displacement(np.zeros((256, 256)), np.zeros((256, 256)), valid=np.ones((128, 128), bool))


class TestMedianShiftPrecision:
    def test_consistent_windows_give_a_small_standard_error(self) -> None:
        shifts = np.random.default_rng(0).normal([0.0, 2.0], 0.3, size=(100, 2))
        assert median_shift_se(shifts) < 0.15

    def test_scattered_windows_give_a_large_one(self) -> None:
        # What a terrain simulation matches on a flat plain: a few windows, anywhere.
        shifts = np.random.default_rng(1).uniform(-60.0, 60.0, size=(12, 2))
        assert median_shift_se(shifts) > 5.0

    def test_overlap_reduces_the_effective_number_of_windows(self) -> None:
        shifts = np.random.default_rng(2).normal(0.0, 1.0, size=(64, 2))
        assert median_shift_se(shifts, overlap_factor=4.0) == pytest.approx(2.0 * median_shift_se(shifts))

    def test_no_windows_is_nan(self) -> None:
        assert np.isnan(median_shift_se(np.empty((0, 2))))


class TestSimulationCheck:
    def test_cos_n_simulation_registers_against_a_correctly_geocoded_image(self) -> None:
        """docs/05 5.1: the peak must sit at (0, 0) when the geocoding is right."""
        rng = np.random.default_rng(3)
        field = ndimage.gaussian_filter(rng.random((512, 512)), 12)
        dem = (field - field.min()) / np.ptp(field) * 800.0  # hilly: ~30 deg median slope
        slope, aspect = horn_slope_aspect(dem, 10.0, 10.0)
        theta_loc = local_incidence_angle(slope, aspect, 39.0, 103.0)
        gamma0_db = simulate_backscatter(theta_loc, exponent=2.0) + rng.normal(0, 0.3, dem.shape)
        good = validate_geocoding(gamma0_db, theta_loc)
        assert good["median_abs_displacement"] < 0.3
        shifted = validate_geocoding(np.roll(gamma0_db, 4, axis=1), theta_loc)
        assert shifted["median_dx"] == pytest.approx(4.0, abs=0.5)

    def test_shadow_and_layover_extremes(self) -> None:
        theta = np.full((4, 4), 30.0)
        shadow = np.zeros((4, 4), bool); shadow[0, 0] = True
        layover = np.zeros((4, 4), bool); layover[3, 3] = True
        db = simulate_backscatter(theta, shadow=shadow, layover=layover)
        assert db[0, 0] == db.min() and db[3, 3] == db.max()
