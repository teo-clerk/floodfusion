"""Copernicus GFM layers on a stack's grid (offline: a fake catalogue and tiny local COGs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from floodfusion.data import gfm
from floodfusion.data.chipping import AnalysisGrid

GRID = AnalysisGrid(epsg=32635, transform=(10.0, 0.0, 433_700.0, 0.0, -10.0, 6_240_640.0), width=40, height=30)
POST = "S1A_IW_GRDH_1SDV_20230408T155612_20230408T155637_048007_05C526_rtc"


def _cog(path: Path, value: int, *, cols: slice = slice(None), rows: slice = slice(None),
         crs: CRS | None = None, transform: Affine | None = None, shape: tuple[int, int] = (15, 20)) -> None:
    data = np.full(shape, gfm.NODATA, dtype=np.uint8)
    data[rows, cols] = value
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", count=1, height=shape[0], width=shape[1],
                       crs=crs or CRS.from_epsg(32635), transform=transform or from_origin(433_700.0, 6_240_640.0,
                                                                                            20.0, 20.0),
                       nodata=gfm.NODATA) as dst:
        dst.write(data, 1)


def _item(name: str, start: str = "20230408T155612", version: str = "GFM v3.1.0") -> dict:
    assets = {key: {"href": f"https://data.eodc.eu/collections/GFM/{name}_{key}.tif"} for _, key in gfm.LAYERS}
    return {"id": f"ENSEMBLE_FLOOD_{start}_VV_EU020M_{name}", "assets": assets,
            "properties": {"datetime": "2023-04-08T15:56:12Z", "processing:version": version,
                           "Equi7Tile": f"EU020M_{name}", "flood_members": {"DLR": True}}}


def _local(tmp_path: Path):
    return lambda href: str(tmp_path / href.rsplit("/", 1)[1].split("GFM_")[-1])


def test_frame_start_reads_the_fifth_field_and_refuses_other_ids() -> None:
    assert gfm.frame_start(POST) == "20230408T155612"
    with pytest.raises(ValueError, match="not a Sentinel-1 product id"):
        gfm.frame_start("S1A_IW_GRDH_1SDV")
    with pytest.raises(ValueError, match="not a Sentinel-1 product id"):
        gfm.frame_start("S1A_IW_GRDH_1SDV_2023-04-08_x")


def test_search_asks_the_catalogue_for_the_frames_window_and_follows_next_links() -> None:
    calls = []
    pages = [{"features": [{"id": "a"}], "links": [{"rel": "next", "href": "https://stac.eodc.eu/api/v1/search?t=2"}]},
             {"features": [{"id": "b"}], "links": []}]

    def get(url, params=None):
        calls.append((url, params))
        return json.dumps(pages[len(calls) - 1]).encode()

    items = gfm.search_items((25.9, 56.1, 26.1, 56.3), ["20230408T155612", "20230408T155543"], get=get)
    assert [i["id"] for i in items] == ["a", "b"]
    assert calls[0][0] == gfm.STAC_SEARCH_URL and calls[1] == ("https://stac.eodc.eu/api/v1/search?t=2", None)
    assert calls[0][1]["collections"] == "GFM"
    assert calls[0][1]["datetime"] == "2023-04-08T15:53:43Z/2023-04-08T15:58:12Z"
    with pytest.raises(ValueError, match="no frame"):
        gfm.search_items((0, 0, 1, 1), [], get=get)


def test_search_stops_on_a_runaway_pagination() -> None:
    page = json.dumps({"features": [], "links": [{"rel": "next", "href": "https://stac.eodc.eu/x"}]}).encode()
    with pytest.raises(RuntimeError, match="pages"):
        gfm.search_items((0, 0, 1, 1), ["20230408T155612"], get=lambda url, params=None: page)


def test_items_are_matched_to_frames_by_start_time_only() -> None:
    items = [_item("A"), _item("B", start="20230408T155543"), _item("C", start="20230408T155637")]
    matched = gfm.match_frames(items, ["20230408T155612", "20230408T155543", "20230409T000000"])
    assert [i["id"][-1] for i in matched["20230408T155612"]] == ["A"]
    assert [i["id"][-1] for i in matched["20230408T155543"]] == ["B"]
    assert matched["20230409T000000"] == []


def test_assets_off_the_eodc_store_are_refused(tmp_path: Path) -> None:
    item = _item("A")
    item["assets"]["ensemble_flood_extent"]["href"] = "https://example.com/flood.tif"
    with pytest.raises(ValueError, match="allowed hosts"):
        gfm.read_layers([item], GRID, source=_local(tmp_path))
    del item["assets"]["ensemble_flood_extent"]
    with pytest.raises(KeyError, match="no asset"):
        gfm.read_layers([item], GRID, source=_local(tmp_path))


def test_layers_mosaic_first_valid_value_wins_and_gaps_stay_nodata(tmp_path: Path) -> None:
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 1, cols=slice(0, 10))                   # left half of the grid
        _cog(tmp_path / f"B_{key}.tif", 2, rows=slice(0, 10))                   # top two thirds, everywhere
    out = gfm.read_layers([_item("A"), _item("B")], GRID, source=_local(tmp_path))
    assert out.shape == (len(gfm.LAYERS), 30, 40) and out.dtype == np.uint8
    for band in out:
        assert (band[:, :20] == 1).all()                                        # A first, 20 m -> 10 m nearest
        assert (band[:20, 20:] == 2).all()                                      # B fills A's gap
        assert (band[20:, 20:] == gfm.NODATA).all()                             # neither covers it


def test_layers_from_another_projection_land_on_the_grid(tmp_path: Path) -> None:
    from rasterio.warp import transform_bounds

    laea = CRS.from_epsg(3035)
    left, _bottom, _right, top = transform_bounds(CRS.from_epsg(32635), laea, *GRID.bounds)
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 7, crs=laea, shape=(40, 40),
             transform=from_origin(left - 100, top + 100, 20.0, 20.0))
    out = gfm.read_layers([_item("A")], GRID, source=_local(tmp_path))
    assert (out == 7).all()


def _event(tmp_path: Path) -> Path:
    d = tmp_path / "events" / "EMSR657_AOI01_t00_00"
    d.mkdir(parents=True)
    (d / "scenes.json").write_text(json.dumps([{"s1_post_id": POST, "s1_post_frames": [POST]}]))
    (d / "event.json").write_text(json.dumps({"epsg": GRID.epsg, "transform": list(GRID.transform),
                                              "width": GRID.width, "height": GRID.height}))
    return d


def test_sidecar_holds_every_layer_and_its_provenance(tmp_path: Path) -> None:
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 1, rows=slice(0, 10))
    event = _event(tmp_path)
    seen = []

    def search(bbox, starts):
        seen.append((bbox, starts))
        return [_item("A"), _item("Z", start="20230408T155637")]

    record = gfm.build_sidecar(event, search=search, source=_local(tmp_path))
    assert record["status"] == "ok" and record["versions"] == ["GFM v3.1.0"]
    assert [i["id"] for i in record["items"]] == [_item("A")["id"]]             # the other frame's item is dropped
    assert record["grid_coverage"] == pytest.approx(20 / 30)
    assert set(record["layer_sha256"]) == {name for name, _ in gfm.LAYERS}
    lon_min, lat_min, lon_max, lat_max = seen[0][0]
    assert 25.92 < lon_min < lon_max < 25.95 and 56.30 < lat_min < lat_max < 56.31
    with rasterio.open(event / gfm.SIDECAR) as src:
        assert src.count == len(gfm.LAYERS) and src.descriptions[0] == "ensemble_flood" and src.nodata == 255
    layers = gfm.read_sidecar(event)
    assert (layers["ensemble_likelihood"][:20] == 1).all() and (layers["list_flood"][20:] == 255).all()
    again = gfm.build_sidecar(event, search=lambda *a: pytest.fail("cached record must not search again"))
    assert again == json.loads((event / gfm.RECORD).read_text())


def test_a_frame_without_a_product_gets_a_record_and_no_raster(tmp_path: Path) -> None:
    event = _event(tmp_path)
    record = gfm.build_sidecar(event, search=lambda bbox, starts: [], source=_local(tmp_path))
    assert record["status"] == "no GFM product for frame start(s) 20230408T155612" and record["items"] == []
    assert not (event / gfm.SIDECAR).exists() and gfm.read_sidecar(event) is None


def test_cli_reports_stacks_and_fails_on_errors(tmp_path: Path, monkeypatch, caplog) -> None:
    event = _event(tmp_path)
    (tmp_path / "events" / "broken").mkdir()
    (tmp_path / "events" / "broken" / "event.json").write_text("{}")
    (tmp_path / "events" / "half_ingested").mkdir()                      # no event.json yet: not swept
    (tmp_path / "events" / "half_ingested" / "scenes.json").write_text("[]")
    monkeypatch.setattr(gfm, "build_sidecar",
                        lambda d, overwrite=False: {"status": "ok"} if d == event else [][0])
    caplog.set_level("INFO", logger="floodfusion.data.gfm")
    assert gfm.main(["--root", str(tmp_path), "--workers", "2"]) == 1
    assert "2 stacks: 1 with GFM, 0 without, 1 errors" in caplog.text            # half_ingested not swept


def test_a_missing_optional_layer_stays_nodata_and_is_recorded(tmp_path: Path) -> None:
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 1, rows=slice(0, 10))
        _cog(tmp_path / f"B_{key}.tif", 2)
    a = _item("A")
    del a["assets"]["exclusion_mask"]                                    # as GFM v4.1.0 item ...T165624 on 2026-04-02
    out = gfm.read_layers([a, _item("B")], GRID, source=_local(tmp_path))
    band = [name for name, _ in gfm.LAYERS].index("exclusion")
    assert (out[band] == 2).all() and (out[0][:20] == 1).all()          # B fills the layer A lacks
    assert gfm.missing_assets(a) == ["exclusion_mask"]
    event = _event(tmp_path)
    record = gfm.build_sidecar(event, search=lambda bbox, starts: [a], source=_local(tmp_path))
    assert record["status"] == "ok" and record["items"][0]["missing_assets"] == ["exclusion_mask"]
    assert "exclusion_mask" not in record["items"][0]["assets"]


def test_no_flood_from_the_first_item_is_not_overwritten_by_flood_from_the_next(tmp_path: Path) -> None:
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 0)                                 # 0 = observed, no flood
        _cog(tmp_path / f"B_{key}.tif", 1)
    out = gfm.read_layers([_item("A"), _item("B")], GRID, source=_local(tmp_path))
    assert (out == 0).all()


def test_only_complete_results_are_cached_and_stale_rasters_go(tmp_path: Path) -> None:
    for _, key in gfm.LAYERS:
        _cog(tmp_path / f"A_{key}.tif", 1)
    event = _event(tmp_path)
    assert gfm.build_sidecar(event, search=lambda b, s: [_item("A")], source=_local(tmp_path))["status"] == "ok"
    (event / gfm.SIDECAR).unlink()                                       # raster lost: the record is not trusted
    assert gfm.build_sidecar(event, search=lambda b, s: [_item("A")], source=_local(tmp_path))["status"] == "ok"
    assert (event / gfm.SIDECAR).is_file()
    gone = gfm.build_sidecar(event, search=lambda b, s: [], source=_local(tmp_path), overwrite=True)
    assert gone["status"].startswith("no GFM product") and not (event / gfm.SIDECAR).exists()
    calls = []
    later = gfm.build_sidecar(event, search=lambda b, s: calls.append(s) or [_item("A")], source=_local(tmp_path))
    assert calls and later["status"] == "ok"                             # a "no product" record is searched again


def test_a_stack_of_two_frames_needs_both(tmp_path: Path) -> None:
    other = "S1A_IW_GRDH_1SDV_20230408T155543_20230408T155612_048007_05C526_rtc"
    event = _event(tmp_path)
    (event / "scenes.json").write_text(json.dumps([{"s1_post_id": POST, "s1_post_frames": [other, POST]}]))
    record = gfm.build_sidecar(event, search=lambda b, s: [_item("A")], source=_local(tmp_path))
    assert record["status"] == "no GFM product for frame start(s) 20230408T155543"


def test_the_default_getter_refuses_a_next_link_off_the_catalogue(monkeypatch) -> None:
    import requests

    class Page:
        status_code, headers = 200, {}

        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def iter_content(self, chunk_size: int):
            yield self.body

    requested = []
    body = json.dumps({"features": [], "links": [{"rel": "next", "href": "https://evil.example/search"}]}).encode()
    monkeypatch.setattr(requests, "get", lambda url, **kw: requested.append(url) or Page(body))
    with pytest.raises(ValueError, match="allowed hosts"):
        gfm.search_items((0, 0, 1, 1), ["20230408T155612"])
    assert requested == [gfm.STAC_SEARCH_URL]                            # the hostile link was never fetched
