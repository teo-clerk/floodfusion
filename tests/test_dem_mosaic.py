"""v1.1 erratum: two GLO-30 tiles without a nodata value must both reach the grid (no zero terrain)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from floodfusion.data import sources
from floodfusion.data.chipping import AnalysisGrid


def _tile(path: Path, lat_top: float, value: float) -> str:
    """A 1° x 1° geographic tile at 0.01° with no nodata declared, like Copernicus GLO-30."""
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=1, height=100, width=100,
                       crs=CRS.from_epsg(4326), transform=from_origin(21.0, lat_top, 0.01, 0.01)) as dst:
        dst.write(np.full((100, 100), value, np.float32), 1)
    return str(path)


def test_a_grid_across_two_dem_tiles_gets_both_heights(tmp_path: Path, monkeypatch) -> None:
    north, south = _tile(tmp_path / "n42.tif", 43.0, 300.0), _tile(tmp_path / "n41.tif", 42.0, 100.0)
    monkeypatch.setattr(sources, "signed", lambda href: href)
    monkeypatch.setattr(sources, "asset_href", lambda item, *keys: item["href"])
    # A UTM 34N grid straddling 42° N (the boundary between the two tiles).
    grid = AnalysisGrid(epsg=32634, transform=(100.0, 0.0, 520_000.0, 0.0, -100.0, 4_654_000.0), width=40, height=80)
    height = sources.read_dem([{"href": north}, {"href": south}], grid)
    assert np.isfinite(height).all()
    assert set(np.unique(np.round(height))) == {100.0, 300.0}           # never 0 where the first tile stops
    lone = sources.read_dem([{"href": north}], grid)
    assert np.isnan(lone).any() and not (lone == 0).any()               # outside the only tile: missing, not 0
