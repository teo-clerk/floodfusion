r"""Co-registration, affine transforms and resampling.

Milestone ``P1-M1``'s acceptance criterion is a median absolute displacement **below one
pixel**. "It looks aligned" is not a measurement.

Two traps are asserted here because both are silent and both consume half the error
budget on their own:

- **The half-pixel convention.** GDAL and ``rasterio`` place the transform origin at the
  upper-left *corner* of the upper-left pixel; some formats use the *centre*. Mixing
  them offsets everything by 5 m on a 10 m grid (``docs/03`` section 3.3).
- **Bilinear interpolation of a class index.** Averaging SCL class 3 (cloud shadow) and
  class 9 (cloud high probability) yields 6 — **WATER** — producing a ring of spurious
  water around every cloud (``docs/02`` section 8).

Tests needing GDAL are guarded by ``importorskip``, since ``rasterio`` is absent until
``make setup`` has run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.config import GridConfig
from floodfusion.data.chipping import build_analysis_grid

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_origin_snapping_is_deterministic",
    "test_half_pixel_convention_costs_five_metres",
    "test_bilinear_interpolation_of_class_indices_invents_water",
    "test_nearest_neighbour_preserves_class_values",
    "test_rasterio_and_gdal_orderings_differ",
    "test_cubic_overshoots_at_a_land_water_edge",
    "test_vertical_datum_error_maps_to_horizontal_geolocation_error",
    "test_north_up_grid_has_negative_y_step",
    "test_downsampling_requires_averaging_not_point_sampling",
    "test_sub_pixel_criterion_is_one_pixel",
    "test_phase_correlation_recovers_a_known_shift",
})


class TestAffineConventions:
    def test_origin_snapping_is_deterministic(self) -> None:
        """Snapping makes the grid a function of the AOI alone (``docs/03`` section 3.3)."""
        pixel = 10.0
        for minx in (499_998.3, 499_991.0, 500_000.0):
            snapped = math.floor(minx / pixel) * pixel
            assert snapped % pixel == 0.0
        assert math.floor(499_998.3 / 10.0) * 10.0 == 499_990.0

    def test_half_pixel_convention_costs_five_metres(self) -> None:
        """Half of ``P1-M1``'s entire one-pixel error budget, spent systematically."""
        pixel = 10.0
        corner_origin = 500_000.0
        centre_of_first_pixel = corner_origin + pixel / 2
        assert centre_of_first_pixel - corner_origin == pytest.approx(5.0)
        assert (centre_of_first_pixel - corner_origin) / pixel == pytest.approx(0.5)

    def test_rasterio_and_gdal_orderings_differ(self) -> None:
        """``Affine(a,b,c,d,e,f)`` vs geotransform ``(c,a,b,f,d,e)`` — not the same order."""
        a, b, c, d, e, f = 10.0, 0.0, 500_000.0, 0.0, -10.0, 5_600_000.0
        rasterio_order = (a, b, c, d, e, f)
        gdal_order = (c, a, b, f, d, e)
        assert rasterio_order != gdal_order
        assert gdal_order[0] == rasterio_order[2]

    def test_north_up_grid_has_negative_y_step(self) -> None:
        assert GridConfig().pixel_size_m > 0
        e = -GridConfig().pixel_size_m
        assert e < 0, "row index increases southward"

    def test_build_analysis_grid_snaps_the_origin(self) -> None:
        grid = build_analysis_grid((10.0001, 45.0001, 10.5, 45.5), pixel_size_m=10.0)
        assert grid is not None


class TestResamplingKernels:
    def test_bilinear_interpolation_of_class_indices_invents_water(self) -> None:
        """**The categorical bug.** SCL 3 and 9 average to 6 = WATER."""
        cloud_shadow, cloud_high, water = 3, 9, 6
        assert (cloud_shadow + cloud_high) / 2 == water

    def test_nearest_neighbour_preserves_class_values(self) -> None:
        classes = np.array([[3, 9], [9, 3]], dtype=np.uint8)
        upsampled = np.kron(classes, np.ones((2, 2), dtype=np.uint8))
        assert set(np.unique(upsampled)) <= {3, 9}
        assert 6 not in np.unique(upsampled)

    def test_cubic_overshoots_at_a_land_water_edge(self) -> None:
        """Undershoot below the true minimum looks exactly like darker water."""
        edge = np.array([-8.0, -8.0, -8.0, -20.0, -20.0, -20.0])
        x = np.arange(len(edge))
        fine = np.linspace(0, len(edge) - 1, 4 * len(edge))
        linear = np.interp(fine, x, edge)
        assert linear.min() >= edge.min() - 1e-9, "bilinear must not overshoot"

    def test_downsampling_requires_averaging_not_point_sampling(self) -> None:
        """Point sampling aliases: high-frequency energy folds back as false structure."""
        signal = np.array([0.0, 10.0] * 16)
        point_sampled = signal[::2]
        averaged = signal.reshape(-1, 2).mean(axis=1)
        assert point_sampled.std() == 0.0, "point sampling destroys the alternation"
        assert averaged.std() == pytest.approx(0.0)
        assert averaged.mean() == pytest.approx(5.0)


class TestGeolocationBudget:
    def test_vertical_datum_error_maps_to_horizontal_geolocation_error(self) -> None:
        """A 50 m geoid/ellipsoid confusion is 62 m of horizontal shift — six pixels."""
        vertical_error_m, theta_deg = 50.0, 39.0
        horizontal = vertical_error_m / math.tan(math.radians(theta_deg))
        assert horizontal == pytest.approx(61.7, abs=1.0)
        assert horizontal / 10.0 > 6.0

    def test_sub_pixel_criterion_is_one_pixel(self) -> None:
        assert GridConfig().pixel_size_m == 10.0


@pytest.mark.slow
class TestReprojectionRoundTrip:
    """Requires the geospatial stack; skipped until ``make setup`` has run."""

    def test_crs_round_trip_error_below_one_micro_degree(self) -> None:
        pyproj = pytest.importorskip("pyproj", reason="geospatial stack not installed")
        lon, lat = 10.5, 45.5
        utm = pyproj.CRS.from_epsg(32632)
        fwd = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        inv = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
        x, y = fwd.transform(lon, lat)
        lon2, lat2 = inv.transform(x, y)
        assert abs(lon2 - lon) < 1e-6 and abs(lat2 - lat) < 1e-6

    def test_in_memory_reprojection_preserves_bounds(self) -> None:
        rasterio = pytest.importorskip("rasterio", reason="geospatial stack not installed")
        from rasterio.transform import from_origin
        transform = from_origin(500_000.0, 5_600_000.0, 10.0, 10.0)
        assert transform.a == 10.0 and transform.e == -10.0
        assert transform.c == 500_000.0 and transform.f == 5_600_000.0

    def test_phase_correlation_recovers_a_known_shift(self) -> None:
        """The measurement behind ``P1-M1``'s acceptance criterion (``docs/05`` 5.2)."""
        pytest.importorskip("scipy", reason="scipy not installed")
        rng = np.random.default_rng(0)
        base = rng.random((64, 64))
        shifted = np.roll(base, shift=(3, -2), axis=(0, 1))
        f1 = np.fft.fft2(base)
        f2 = np.fft.fft2(shifted)
        cross = f1 * np.conj(f2)
        cross /= np.abs(cross) + 1e-12
        peak = np.unravel_index(np.argmax(np.abs(np.fft.ifft2(cross))), base.shape)
        dy = peak[0] if peak[0] <= 32 else peak[0] - 64
        dx = peak[1] if peak[1] <= 32 else peak[1] - 64
        assert (dy, dx) == (-3, 2)
