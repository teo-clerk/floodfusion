r"""Range--Doppler backward geocoding and area-based RTC on a simulated orbit (Route A core).

The scene: a 3 x 3 km grid at 45 N, 10 E (UTM 32N), a straight-line orbit 700 km up and
560 km west of it, flying north at 7 km/s and looking right (east) — incidence ~39 deg.
"""

from __future__ import annotations

import numpy as np
import pytest

pyproj = pytest.importorskip("pyproj")

from floodfusion.data.geocoding import (
    OrbitStateVectors,
    SlantRangeGeometry,
    SRGRGeometry,
    _enu_basis,
    ecef_to_geodetic,
    geodetic_to_ecef,
    terrain_correct,
    zero_doppler_time,
)

LAT0, LON0, H, D, SPEED = 45.0, 10.0, 700e3, 560e3, 7000.0
P0 = geodetic_to_ecef(LAT0, LON0, 0.0)
EAST, NORTH, UP = _enu_basis(np.array(LAT0), np.array(LON0))
R0 = float(np.hypot(H, D))


def _orbit(spacing_s: float = 2.0) -> OrbitStateVectors:
    t = np.arange(-30.0, 30.0 + spacing_s, spacing_s)
    pos = P0 + (-D) * EAST + H * UP + (SPEED * t)[:, None] * NORTH
    return OrbitStateVectors(t, pos, np.repeat((SPEED * NORTH)[None], t.size, 0))


GEOMETRY = SlantRangeGeometry(first_line_time=-0.5, line_interval=0.002, near_range_m=R0 - 2500, range_spacing_m=10.0)


def _grid(pixel: float = 5.0, half_m: float = 1500.0):
    x0, y0 = pyproj.Transformer.from_crs(4326, 32632, always_xy=True).transform(LON0, LAT0)
    n = int(2 * half_m / pixel)
    return (pixel, 0.0, x0 - half_m, 0.0, -pixel, y0 + half_m), np.zeros((n, n))


class TestGeodesy:
    def test_ecef_round_trip(self) -> None:
        lat, lon, h = ecef_to_geodetic(geodetic_to_ecef(52.3, -4.1, 812.0))
        assert lat == pytest.approx(52.3, abs=1e-9) and lon == pytest.approx(-4.1, abs=1e-9)
        assert h == pytest.approx(812.0, abs=1e-4)

    def test_equator_is_the_semi_major_axis(self) -> None:
        assert np.allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [6_378_137.0, 0.0, 0.0])


class TestOrbit:
    def test_hermite_reproduces_the_state_vectors(self) -> None:
        orbit = _orbit()
        pos, vel, _ = orbit.interpolate(orbit.times[3])
        assert np.allclose(pos, orbit.positions[3]) and np.allclose(vel, orbit.velocities[3])

    def test_straight_line_is_interpolated_exactly(self) -> None:
        orbit = _orbit(spacing_s=10.0)
        pos, _, acc = orbit.interpolate(np.array(3.7))
        assert np.allclose(pos, P0 + (-D) * EAST + H * UP + SPEED * 3.7 * NORTH, atol=1e-6)
        assert np.allclose(acc, 0.0, atol=1e-9)

    def test_rejects_unsorted_times(self) -> None:
        with pytest.raises(ValueError):
            OrbitStateVectors(np.array([1.0, 0.0]), np.zeros((2, 3)), np.zeros((2, 3)))


class TestZeroDoppler:
    def test_closest_approach_of_the_scene_centre_is_t0(self) -> None:
        assert zero_doppler_time(_orbit(), P0[None])[0] == pytest.approx(0.0, abs=1e-6)

    def test_targets_north_are_imaged_later(self) -> None:
        north_target = P0 + 1400.0 * NORTH
        t = zero_doppler_time(_orbit(), north_target[None])[0]
        assert t == pytest.approx(1400.0 / SPEED, rel=1e-6)

    def test_satisfies_the_zero_doppler_condition(self) -> None:
        orbit = _orbit()
        targets = P0 + np.array([[300.0, -800.0, 0.0], [-1200.0, 500.0, 10.0]]) @ np.stack([EAST, NORTH, UP])
        t = zero_doppler_time(orbit, targets)
        s, v, _ = orbit.interpolate(t)
        assert np.allclose(np.einsum("ij,ij->i", v, targets - s), 0.0, atol=1e-3)


class TestBackwardGeocoding:
    def test_every_in_swath_cell_is_filled_exactly_once(self) -> None:
        """Output-first: one interpolation per output cell, no holes (docs/04 section 5.3)."""
        transform, dem = _grid()
        out, diag = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632,
                                    rtc=None, return_diagnostics=True)
        inside = (diag["line"] >= 0) & (diag["line"] <= 499) & (diag["pixel"] >= 0) & (diag["pixel"] <= 499)
        assert inside.any()
        assert np.isfinite(out[inside]).all() and np.isnan(out[~inside]).all()

    def test_point_target_lands_in_the_right_cell(self) -> None:
        orbit = _orbit()
        transform, dem = _grid(pixel=10.0)
        # A bright scatterer on the ground 400 m east and 600 m north of the centre.
        target = P0 + 400.0 * EAST + 600.0 * NORTH
        lat, lon, _ = ecef_to_geodetic(target)
        t = zero_doppler_time(orbit, target[None])[0]
        s, _, _ = orbit.interpolate(t)
        line, pixel = GEOMETRY.to_image(t, np.linalg.norm(target - s))
        image = np.full((500, 500), 1e-3)
        image[int(round(line)), int(round(pixel))] = 100.0
        out = terrain_correct(image, GEOMETRY, orbit, dem, transform, 32632, rtc=None)
        row, col = np.unravel_index(np.nanargmax(out), out.shape)
        x, y = pyproj.Transformer.from_crs(4326, 32632, always_xy=True).transform(lon, lat)
        expected_col = (x - transform[2]) / transform[0]
        expected_row = (y - transform[5]) / transform[4]
        assert abs(col + 0.5 - expected_col) <= 1.5 and abs(row + 0.5 - expected_row) <= 1.5

    def test_flat_terrain_area_rtc_is_beta0_tan_theta(self) -> None:
        """Small (2011) on flat ground reduces to gamma0 = beta0 tan(theta)."""
        transform, dem = _grid()
        out, diag = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632,
                                    rtc="area", return_diagnostics=True)
        ratio = out / np.tan(np.radians(diag["theta_ell_deg"]))
        interior = ratio[50:-50, 50:-50]
        assert np.nanmedian(interior) == pytest.approx(1.0, abs=0.01)
        assert np.nanpercentile(interior, 95) < 1.05 and np.nanpercentile(interior, 5) > 0.95

    def test_cos_rtc_matches_area_rtc_on_flat_ground(self) -> None:
        transform, dem = _grid()
        cos = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632, rtc="cos")
        area = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632, rtc="area")
        assert np.nanmedian(cos[50:-50, 50:-50] / area[50:-50, 50:-50]) == pytest.approx(1.0, abs=0.01)

    def test_slope_facing_the_sensor_is_darkened_by_rtc(self) -> None:
        """Foreshortening brightens a facing slope; RTC divides by the larger contributing area."""
        transform, dem = _grid()
        cols = np.arange(dem.shape[1]) * transform[0]
        # The sensor is WEST of the scene, so a facing slope RISES eastward (21.8 deg).
        facing = np.broadcast_to(np.clip(cols - 1500.0, 0, None) * 0.4, dem.shape).copy()
        flat = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632, rtc="area")
        sloped, diag = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), facing, transform, 32632,
                                       rtc="area", return_diagnostics=True)
        zone = (slice(100, -100), slice(350, 550))
        theta_loc = np.nanmedian(diag["theta_loc_deg"][zone])
        assert theta_loc == pytest.approx(np.nanmedian(diag["theta_ell_deg"][zone]) - 21.8, abs=0.5)
        assert np.nanmedian(sloped[zone]) < np.nanmedian(flat[zone])

    def test_geoid_offset_shifts_geolocation(self) -> None:
        """A 50 m height error moves the image by ~50/tan(theta) = 62 m (docs/03 2.3)."""
        transform, dem = _grid()
        _, a = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632,
                               rtc=None, return_diagnostics=True)
        _, b = terrain_correct(np.ones((500, 500)), GEOMETRY, _orbit(), dem, transform, 32632,
                               rtc=None, geoid_undulation_m=50.0, return_diagnostics=True)
        shift_m = np.nanmedian(a["slant_range"] - b["slant_range"]) / np.sin(np.radians(39.0))
        assert shift_m == pytest.approx(50.0 / np.tan(np.radians(38.66)), rel=0.1)

    def test_rejects_unknown_rtc(self) -> None:
        transform, dem = _grid()
        with pytest.raises(ValueError, match="rtc"):
            terrain_correct(np.ones((5, 5)), GEOMETRY, _orbit(), dem, transform, 32632, rtc="flat")


class TestSRGR:
    @staticmethod
    def _geometry(drift_per_s: float = 0.0) -> SRGRGeometry:
        # g = 1.6 (R - sr0) + 1e-6 (R - sr0)^2 and its exact inverse to first order.
        return SRGRGeometry(
            first_line_time=0.0, line_interval=0.002, ground_spacing_m=10.0,
            poly_times=(0.0, 10.0), sr0_m=(R0 - 2000.0, R0 - 2000.0),
            srgr_coefficients=((0.0, 1.6, 1e-6), (10.0 * drift_per_s, 1.6, 1e-6)),
            gr0_m=(0.0, 0.0), grsr_coefficients=((R0 - 2000.0, 0.625), (R0 - 2000.0, 0.625)),
        )

    def test_slant_to_ground_polynomial_is_applied(self) -> None:
        geom = self._geometry()
        dr = np.array([0.0, 1000.0, 2500.0])
        _, pixel = geom.to_image(np.zeros(3), R0 - 2000.0 + dr)
        assert np.allclose(pixel * 10.0, 1.6 * dr + 1e-6 * dr**2)
        assert np.allclose(geom.slant_range_spacing(np.array([0.0])), 6.25)

    def test_polynomials_are_interpolated_in_azimuth_time(self) -> None:
        """Coefficients drift along track; mid-way between two the offset is halved."""
        geom = self._geometry(drift_per_s=8.0)
        _, start = geom.to_image(np.array([0.0]), np.array([R0]))
        _, mid = geom.to_image(np.array([5.0]), np.array([R0]))
        _, end = geom.to_image(np.array([10.0]), np.array([R0]))
        assert (end - start)[0] * 10.0 == pytest.approx(80.0)
        assert (mid - start)[0] * 10.0 == pytest.approx(40.0)

    def test_rejects_mismatched_polynomial_lists(self) -> None:
        with pytest.raises(ValueError):
            SRGRGeometry(0.0, 0.002, 10.0, (0.0,), (1.0,), (), (0.0,), ((1.0,),))
