"""The shared per-grid assembly: grid expansion and terrain computed with context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from floodfusion.data.assembly import expand_grid, terrain_on_grid
from floodfusion.data.chipping import build_analysis_grid
from floodfusion.data.s1_annotation import parse_annotation

ANNOTATION = Path(__file__).parent / "fixtures" / "s1a_iw_grd_vv_20231107t170700_annotation_trimmed.xml"
CHIP = (10.200, 44.400, 10.215, 44.412)       # ~1.2 x 1.3 km inside the fixture's frame
DEM_BOUNDS = (10.17, 44.38, 10.25, 44.43)
WALL_LON = (10.1975, 10.1980)                  # a 600 m wall ~200 m west of the chip


class TestExpandGrid:
    def test_margin_keeps_pixels_aligned(self) -> None:
        grid = build_analysis_grid(CHIP, 10.0)
        big = expand_grid(grid, 7)
        assert (big.width, big.height) == (grid.width + 14, grid.height + 14)
        assert big.pixel_centre(7, 7) == pytest.approx(grid.pixel_centre(0, 0))
        x0, y0, x1, y1 = grid.bounds
        assert big.bounds == pytest.approx((x0 - 70, y0 - 70, x1 + 70, y1 + 70))

    def test_zero_margin_is_identity_and_negative_is_refused(self) -> None:
        grid = build_analysis_grid(CHIP, 10.0)
        assert expand_grid(grid, 0) is grid
        with pytest.raises(ValueError, match="margin"):
            expand_grid(grid, -1)


@pytest.fixture(scope="module")
def wall_dem(tmp_path_factory) -> list[dict]:
    """Flat terrain at 20 m with a 600 m north-south wall just outside the chip's west edge."""
    w, h = 800, 500
    lon = np.linspace(DEM_BOUNDS[0], DEM_BOUNDS[2], w, endpoint=False) + 0.5 * (DEM_BOUNDS[2] - DEM_BOUNDS[0]) / w
    dem = np.full((h, w), 20.0, np.float32)
    dem[:, (lon >= WALL_LON[0]) & (lon <= WALL_LON[1])] = 620.0
    path = tmp_path_factory.mktemp("dem") / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(*DEM_BOUNDS, w, h)) as dst:
        dst.write(dem, 1)
    return [{"id": "dem", "collection": "cop-dem-glo-30", "bbox": list(DEM_BOUNDS),
             "properties": {"datetime": "2021-04-22T00:00:00Z"}, "assets": {"data": {"href": str(path)}}}]


class TestTerrainWithContext:
    """Ascending pass, sensor to the WSW (phi ~ 256 deg): the wall's shadow falls east, into the chip."""

    def test_shadow_cast_from_outside_the_grid_needs_the_margin(self, wall_dem) -> None:
        annotation = parse_annotation(ANNOTATION.read_bytes())
        grid = build_analysis_grid(CHIP, 10.0)
        phi = annotation.look_azimuth_deg
        bare, theta = terrain_on_grid(wall_dem, grid, [annotation], phi, margin_px=0)
        context, theta_ctx = terrain_on_grid(wall_dem, grid, [annotation], phi, margin_px=64)

        assert bare["shadow"].shape == context["shadow"].shape == (grid.height, grid.width)
        np.testing.assert_allclose(theta, theta_ctx, atol=1e-6)  # cropping keeps the geometry
        west = slice(0, 10)
        assert not bare["shadow"][:, west].any()           # the wall is not in the grid
        assert context["shadow"][:, west].mean() > 0.9      # with context, its shadow is
        assert not context["shadow"][:, -10:].any()         # and it ends well before the east edge


class TestOpticalAlignment:
    """Measured on the real archive: S2 L2A of 2018-2019 sits ~2 px off the RTC geometry."""

    def test_shift_pixels_moves_whole_pixels_and_fills(self) -> None:
        from floodfusion.data.assembly import shift_pixels

        a = np.arange(20.0).reshape(4, 5)
        out = shift_pixels(a, 1, -2, np.nan)
        assert out[0, 2] == a[1, 0] and out[2, 4] == a[3, 2]
        assert np.isnan(out[3]).all() and np.isnan(out[:, :2]).all()
        stack = np.stack([a, a + 100])
        assert shift_pixels(stack, 0, 1, -1.0)[1, 0, 0] == 101.0
        assert (shift_pixels(a, 9, 0, 0.0) == 0).all()

    def _scene(self, dy: int, dx: int):
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(3)
        field = gaussian_filter(rng.normal(size=(320, 320)), 3)
        sar_db = -12.0 + 4.0 * field
        moved = np.roll(field, (dy, dx), axis=(0, 1))                  # moved(r, c) = field(r - dy, c - dx)
        # SWIR carries the shared (water/land) structure; registration is done on it.
        rho = np.stack([np.full_like(moved, 0.08), 0.25 + 0.05 * moved, 0.20 + 0.05 * moved])
        return sar_db, rho, np.ones_like(moved)

    def test_a_two_pixel_offset_is_measured_and_removed(self) -> None:
        from floodfusion.data.assembly import _sar_optical_offset, align_optical_to_sar

        sar, rho, quality = self._scene(0, 2)
        label = np.zeros_like(quality)
        rho2, q2, label2, inside2, info = align_optical_to_sar(sar, rho, quality, label=label,
                                                               inside_aoi=np.ones_like(quality, bool))
        assert info["applied_shift_px"] == [0, 2] and info["label_shifted"] and np.isnan(label2[:, -2:]).all()
        assert np.isnan(rho2[2][:, -2:]).all() and (q2[:, -2:] == 0).all() and not inside2[:, -2:].any()
        residual = _sar_optical_offset(sar, np.nan_to_num(rho2, nan=0.25))
        assert residual["median_shift_px"] < 0.5

    def test_sub_threshold_offsets_are_left_alone(self) -> None:
        from floodfusion.data.assembly import align_optical_to_sar

        sar, rho, quality = self._scene(0, 0)
        rho2, q2, label2, inside2, info = align_optical_to_sar(sar, rho, quality)
        assert info["applied_shift_px"] == [0, 0] and rho2 is rho and q2 is quality
        assert label2 is None and inside2 is None

    def test_pooled_shift_uses_every_window_and_ignores_too_little_evidence(self) -> None:
        from floodfusion.data.assembly import pooled_optical_shift

        rng = np.random.default_rng(0)
        good = [np.column_stack([rng.normal(0.1, 0.4, 30), rng.normal(2.2, 0.4, 30)]) for _ in range(4)]
        noisy_grid = np.array([[5.0, 0.0], [4.8, 0.2]])           # two outlier windows on one grid
        pooled = pooled_optical_shift([*good, noisy_grid, np.empty((0, 2))])
        assert pooled["shift_px"] == [0, 2] and pooled["n_windows"] == 122 and pooled["n_grids"] == 5
        assert pooled_optical_shift([np.array([[0.0, 3.0]])])["shift_px"] == [0, 0]          # < 5 windows
        assert pooled_optical_shift([np.column_stack([np.zeros(20), np.full(20, 0.6)])])["shift_px"] == [0, 0]

    def test_a_fixed_shift_overrides_the_local_estimate(self) -> None:
        from floodfusion.data.assembly import align_optical_to_sar

        sar, rho, quality = self._scene(0, 0)
        rho2, _, _, _, info = align_optical_to_sar(sar, rho, quality, fixed_shift_px=(0, 2))
        assert info["applied_shift_px"] == [0, 2] and info["source"] == "pooled"
        assert np.isnan(rho2[1][:, -2:]).all()
