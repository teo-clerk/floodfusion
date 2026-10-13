"""Sidecar layers beside the stack (v1.2): windows, fixed normalisation, architecture registry, linking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from floodfusion.config import NormalizationStats
from floodfusion.data import sidecars


def _raster(path: Path, data: np.ndarray, nodata: float | None = None) -> None:
    with rasterio.open(path, "w", driver="GTiff", dtype=str(data.dtype), count=data.shape[0], height=data.shape[1],
                       width=data.shape[2], crs=CRS.from_epsg(32634), transform=from_origin(0, 100, 10, 10),
                       nodata=nodata) as dst:
        dst.write(data)


def test_windows_are_boundless_and_nodata_becomes_nan(tmp_path: Path) -> None:
    data = np.arange(2 * 10 * 10, dtype=np.float32).reshape(2, 10, 10)
    data[0, 0, 0] = -9999
    _raster(tmp_path / "s.tif", data, nodata=-9999)
    with rasterio.open(tmp_path / "s.tif") as src:
        w = sidecars.read_sidecar_window(src, (0, 0, 4, 4), 2, 2)
    assert w.shape == (2, 8, 8) and np.isnan(w[:, :2]).all()           # context beyond the raster edge
    assert np.isnan(w[0, 2, 2]) and w[1, 2, 2] == data[1, 0, 0]         # nodata: NaN; band order kept


def test_normalisation_uses_fixed_statistics_only() -> None:
    norm = NormalizationStats()
    raw = np.stack([np.full((3, 3), -15.0), np.full((3, 3), -22.0)])
    post = np.full((3, 3), -20.0)
    out = sidecars.normalise_sidecar("seasonal", raw, vv_post_db=post, norm=norm)
    assert out.shape == (3, 3, 3)
    m, s = norm.sar_mean_db, norm.sar_std_db
    assert out[0, 0, 0] == pytest.approx((-15.0 - m[2]) / s[2]) and out[1, 0, 0] == pytest.approx((-22.0 - m[3]) / s[3])
    assert out[2, 0, 0] == pytest.approx((-5.0 - m[4]) / s[4])          # post - seasonal: a darkening of 5 dB
    assert np.isnan(sidecars.normalise_sidecar("seasonal", raw, vv_post_db=None, norm=norm)[2]).all()
    aef = sidecars.normalise_sidecar("aef", np.full((8, 2, 2), 127.0), vv_post_db=None, norm=norm)
    assert aef.shape == (8, 2, 2) and np.allclose(aef, 2.0)                # int8 127 = a PC of 2

    from floodfusion.data.alphaearth import AEF_SCALE
    assert sidecars.AEF_STORE_SCALE == AEF_SCALE
    assert sidecars.normalise_sidecar("dem", np.full((1, 2, 2), 250.0), vv_post_db=None, norm=norm)[0, 0, 0] == 250.0
    with pytest.raises(KeyError):
        sidecars.normalise_sidecar("ndvi", raw, vv_post_db=None, norm=norm)


def test_architectures_name_their_sidecars_and_others_read_none() -> None:
    assert sidecars.sidecars_for("fusion_crossattn_ctx") == ("seasonal", "aef")
    assert sidecars.sidecars_for("terramind_base_fair") == ("dem",)
    assert sidecars.sidecars_for("fusion_crossattn") == ()
    assert sidecars.channels(("seasonal", "aef")) == 11


def _event(root: Path, name: str, transform: list[float]) -> Path:
    d = root / "events" / name
    d.mkdir(parents=True)
    (d / "event.json").write_text(json.dumps({"epsg": 32634, "transform": transform, "width": 10, "height": 10}))
    return d


def test_unchanged_layers_are_linked_only_onto_identical_grids(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    a = _event(old, "a", [10, 0, 0, 0, -10, 100])
    (a / "seasonal.tif").write_bytes(b"s")
    _event(new, "a", [10, 0, 0, 0, -10, 100])
    b = _event(old, "b", [10, 0, 0, 0, -10, 100])
    (b / "seasonal.tif").write_bytes(b"s")
    _event(new, "b", [10, 0, 5, 0, -10, 100])                          # shifted grid: never linked
    _event(new, "c", [10, 0, 0, 0, -10, 100])                          # no old stack
    counts = sidecars.link_from(old, new, ["seasonal.tif", "gfm.tif"])
    assert counts == {"linked": 1, "present": 0, "grid_differs": 1, "absent": 1}
    assert (new / "events" / "a" / "seasonal.tif").stat().st_ino == (a / "seasonal.tif").stat().st_ino
    assert sidecars.link_from(old, new, ["seasonal.tif"])["present"] == 1    # a re-run changes nothing
