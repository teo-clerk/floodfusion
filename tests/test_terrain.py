r"""Terrain derivatives and the exclusion masks.

The specification for :mod:`floodfusion.masks`. Milestone ``P1-M1``.

The headline test is
:meth:`TestHornSlope.test_geographic_crs_produces_absurd_slopes`, which reproduces the
metric-CRS bug of ``docs/03`` section 4.2: computing Horn slope on an EPSG:4326 grid
divides metres by *degrees*, and at 45 degrees N a 5 m rise across one 1-arcsecond pixel
yields ``arctan(1.8e4) = 89.997`` degrees. Every pixel becomes a cliff.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.config import SensorConstants, ValidityFlag
from floodfusion.masks import (
    build_validity_mask,
    cast_shadow_and_layover,
    hand,
    horn_slope_aspect,
    local_incidence_angle,
)

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_geographic_crs_produces_absurd_slopes",
    "test_longitude_metres_per_degree_shrinks_with_latitude",
    "test_layover_and_shadow_thresholds_at_mid_swath",
    "test_local_incidence_formula_limiting_cases",
    "test_aspect_must_be_decomposed_into_sin_and_cos",
})


class TestHornSlope:
    def test_geographic_crs_produces_absurd_slopes(self) -> None:
        """**The metric-CRS bug**, reproduced arithmetically (``docs/03`` section 4.2)."""
        rise_m, arcsecond_deg = 5.0, 1.0 / 3600.0
        slope_wrong = math.degrees(math.atan(rise_m / arcsecond_deg))
        assert slope_wrong > 89.99, "degrees in the denominator make everything vertical"

        pixel_m = 30.0
        slope_right = math.degrees(math.atan(rise_m / pixel_m))
        assert 5.0 < slope_right < 15.0

    def test_longitude_metres_per_degree_shrinks_with_latitude(self) -> None:
        """The anisotropy variant: 111 320 m/deg is right for latitude, wrong for longitude."""
        lat_m_per_deg = 111_320.0
        for lat, expected in ((0.0, 111_320.0), (45.0, 78_710.0), (60.0, 55_660.0)):
            assert lat_m_per_deg * math.cos(math.radians(lat)) == pytest.approx(expected, rel=0.01)

    def test_flat_dem_has_zero_slope(self, synthetic_dem: np.ndarray) -> None:
        flat = np.full_like(synthetic_dem, 100.0)
        slope, _ = horn_slope_aspect(flat, 30.0, 30.0)
        assert np.allclose(slope, 0.0, atol=1e-9)

    def test_constant_gradient_recovers_the_analytic_slope(self) -> None:
        """A 10 % grade must read as arctan(0.1) = 5.71 degrees."""
        ny, nx = 32, 32
        pixel = 30.0
        z = np.arange(ny, dtype=np.float64)[:, None] * (0.1 * pixel) * np.ones((1, nx))
        slope, _ = horn_slope_aspect(z, pixel, pixel)
        interior = slope[2:-2, 2:-2]
        assert np.allclose(interior, math.degrees(math.atan(0.1)), atol=0.5)

    def test_slope_is_bounded(self, synthetic_dem: np.ndarray) -> None:
        slope, aspect = horn_slope_aspect(synthetic_dem, 30.0, 30.0)
        assert slope.min() >= 0.0 and slope.max() <= 90.0
        assert aspect.min() >= 0.0 and aspect.max() < 360.0

    def test_output_shape_matches_input(self, synthetic_dem: np.ndarray) -> None:
        slope, aspect = horn_slope_aspect(synthetic_dem, 30.0, 30.0)
        assert slope.shape == aspect.shape == synthetic_dem.shape

    def test_non_square_pixels_are_honoured(self, synthetic_dem: np.ndarray) -> None:
        """GLO-30 has non-square pixels in geographic coordinates above 50 degrees N."""
        square, _ = horn_slope_aspect(synthetic_dem, 30.0, 30.0)
        stretched, _ = horn_slope_aspect(synthetic_dem, 45.0, 30.0)
        assert not np.allclose(square, stretched)

    def test_aspect_must_be_decomposed_into_sin_and_cos(self) -> None:
        """359 and 1 degrees are adjacent directions but numerically maximally distant."""
        a, b = math.radians(359.0), math.radians(1.0)
        assert abs(359.0 - 1.0) == 358.0
        assert math.hypot(math.sin(a) - math.sin(b), math.cos(a) - math.cos(b)) < 0.05


class TestLocalIncidenceAngle:
    def test_local_incidence_formula_limiting_cases(self) -> None:
        r"""cos(theta_loc) = cos(a)cos(t) + sin(a)sin(t)cos(phi - A)."""
        def theta_loc(alpha, theta, phi, aspect):
            a, t = math.radians(alpha), math.radians(theta)
            d = math.radians(phi - aspect)
            return math.degrees(math.acos(math.cos(a) * math.cos(t)
                                          + math.sin(a) * math.sin(t) * math.cos(d)))
        assert theta_loc(0.0, 39.0, 280.0, 0.0) == pytest.approx(39.0, abs=1e-9)
        assert theta_loc(25.0, 39.0, 280.0, 280.0) == pytest.approx(14.0, abs=1e-6)
        assert theta_loc(25.0, 39.0, 280.0, 100.0) == pytest.approx(64.0, abs=1e-6)

    def test_flat_terrain_returns_the_ellipsoid_angle(self) -> None:
        slope = np.zeros((8, 8))
        aspect = np.zeros((8, 8))
        out = local_incidence_angle(slope, aspect, 39.0, 280.0)
        assert np.allclose(out, 39.0, atol=1e-6)

    def test_result_is_bounded(self, synthetic_dem: np.ndarray) -> None:
        slope, aspect = horn_slope_aspect(synthetic_dem, 30.0, 30.0)
        out = local_incidence_angle(slope, aspect, 39.0, 280.0)
        assert out.min() >= 0.0 and out.max() <= 180.0

    def test_accepts_a_per_pixel_incidence_field(self, synthetic_dem: np.ndarray) -> None:
        """Incidence varies 29-46 degrees across the swath; a scalar discards that."""
        slope, aspect = horn_slope_aspect(synthetic_dem, 30.0, 30.0)
        field = np.linspace(29.1, 46.0, synthetic_dem.shape[1])[None, :] * np.ones_like(synthetic_dem)
        out = local_incidence_angle(slope, aspect, field, 280.0)
        assert out.shape == synthetic_dem.shape


class TestLayoverAndShadow:
    def test_layover_and_shadow_thresholds_at_mid_swath(self) -> None:
        """At theta = 39 deg: layover above 39 deg facing, shadow above 51 deg away."""
        theta = 39.0
        assert theta == pytest.approx(39.0)
        assert 90.0 - theta == pytest.approx(51.0)

    def test_masks_are_boolean_and_shaped(self, synthetic_dem: np.ndarray) -> None:
        shadow, layover = cast_shadow_and_layover(synthetic_dem, 39.0, 280.0, 30.0)
        assert shadow.dtype == bool and layover.dtype == bool
        assert shadow.shape == layover.shape == synthetic_dem.shape

    def test_flat_terrain_has_no_shadow_or_layover(self) -> None:
        flat = np.full((32, 48), 100.0)
        shadow, layover = cast_shadow_and_layover(flat, 39.0, 280.0, 30.0)
        assert not shadow.any() and not layover.any()

    def test_a_ridge_casts_shadow_onto_flat_ground(self, synthetic_dem: np.ndarray) -> None:
        """**Cast shadow, not just the local test.**

        Flat ground behind a ridge has alpha = 0, so theta_loc = theta and it PASSES the
        local test — while receiving no illumination at all (``docs/03`` section 5.3).
        """
        shadow, _ = cast_shadow_and_layover(synthetic_dem, 39.0, 270.0, 30.0)
        assert shadow.any(), "a 300 m ridge must cast shadow on the terrain behind it"

    def test_ascending_and_descending_shadows_differ(self, synthetic_dem: np.ndarray) -> None:
        """Look azimuth flips ~180 deg, so every facing slope now faces away."""
        desc, _ = cast_shadow_and_layover(synthetic_dem, 39.0, 280.0, 30.0)
        asc, _ = cast_shadow_and_layover(synthetic_dem, 39.0, 100.0, 30.0)
        assert not np.array_equal(desc, asc)


class TestHand:
    def test_shape_and_non_negativity(self, synthetic_dem: np.ndarray) -> None:
        out = hand(synthetic_dem, accumulation_threshold=50.0, pixel_size_m=30.0)
        assert out.shape == synthetic_dem.shape
        assert np.nanmin(out) >= -1e-6, "HAND is a height ABOVE drainage"

    def test_threshold_changes_the_result(self, synthetic_dem: np.ndarray) -> None:
        """The accumulation threshold is a real parameter, not a constant."""
        sparse = hand(synthetic_dem, 500.0, 30.0)
        dense = hand(synthetic_dem, 10.0, 30.0)
        assert not np.allclose(sparse, dense)


class TestValidityMask:
    def test_flags_compose_into_a_uint8_band(self, synthetic_dem: np.ndarray) -> None:
        shape = synthetic_dem.shape
        zeros = np.zeros(shape, dtype=bool)
        mask = build_validity_mask(zeros, zeros, zeros, zeros,
                                   np.zeros(shape), slope_max_deg=10.0)
        assert mask.dtype == np.uint8
        assert (mask == int(ValidityFlag.VALID)).all()

    def test_shadow_sets_only_the_shadow_bit(self, synthetic_dem: np.ndarray) -> None:
        """Bit flags keep exclusion CAUSES separable for error analysis."""
        shape = synthetic_dem.shape
        zeros = np.zeros(shape, dtype=bool)
        shadow = np.zeros(shape, dtype=bool); shadow[0, 0] = True
        mask = build_validity_mask(zeros, shadow, zeros, zeros,
                                   np.zeros(shape), slope_max_deg=10.0)
        assert mask[0, 0] & int(ValidityFlag.SHADOW)
        assert not mask[0, 0] & int(ValidityFlag.LAYOVER)

    def test_steep_slope_is_excluded(self, synthetic_dem: np.ndarray) -> None:
        shape = synthetic_dem.shape
        zeros = np.zeros(shape, dtype=bool)
        slope = np.full(shape, 30.0)
        mask = build_validity_mask(zeros, zeros, zeros, zeros, slope, slope_max_deg=10.0)
        assert (mask & int(ValidityFlag.STEEP_SLOPE)).all()
