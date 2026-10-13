"""v1.1 baseline context: the event-level threshold fallback and the seasonal reference."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from floodfusion.baseline_runner import SceneLayers, load_scene, threshold_surface
from floodfusion.config import BaselineConfig, load_config
from floodfusion.data.chipping import AnalysisGrid, write_event_stack
from floodfusion.data.encoding import N_CHIP_BANDS, ChipBand
from floodfusion.data.seasonal import MIN_SCENES, anniversary_window, read_seasonal, seasonal_reference

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "row02_baseline_adaptive.yaml")
GRID = AnalysisGrid(epsg=32633, transform=(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_900_000.0), width=16, height=16)


def _scene(vv: np.ndarray) -> SceneLayers:
    z = np.zeros(vv.shape, dtype=bool)
    return SceneLayers(vv_post_db=vv, vv_pre_db=vv + 5.0, mndwi=None, optical_valid=None, slope=np.zeros(vv.shape),
                       hand=np.zeros(vv.shape), layover=z, shadow=z, permanent=z, label=z, scored=~z)


class TestEventFallback:
    def test_a_scene_without_an_accepted_tile_takes_its_events_threshold(self) -> None:
        flat = _scene(np.random.default_rng(0).normal(-8.0, 0.5, (64, 64)))   # unimodal: no tile accepted
        surface, info = threshold_surface(flat, CFG, fallback_db=-17.5)
        assert info["fallback"] == "event" and np.all(surface == -17.5)
        surface, info = threshold_surface(flat, CFG)
        assert info["fallback"] == "global_otsu" and np.all(surface == info["otsu_db"])

    def test_an_accepted_scene_reports_its_median_threshold(self) -> None:
        rng = np.random.default_rng(1)
        stripes = (np.arange(256)[:, None] % 64) < 24      # every 64 px tile: 37.5 % water, bimodal
        vv = np.where(stripes, rng.normal(-22.0, 1.0, (256, 256)), rng.normal(-8.0, 1.0, (256, 256)))
        _, info = threshold_surface(_scene(vv), replace(CFG, baseline=replace(CFG.baseline, tile_size=64)))
        assert "fallback" not in info and -22.0 < info["accepted_median_db"] < -8.0

    @pytest.mark.parametrize(("field", "value"), [("reference", "yesterday"), ("threshold_fallback", "neighbour")])
    def test_options_are_validated(self, field, value) -> None:
        with pytest.raises(ValueError, match=field):
            BaselineConfig(**{field: value})


def _stack(tmp_path: Path, *, pre_db: float = -9.0, post_db: float = -20.0) -> Path:
    stack = np.zeros((N_CHIP_BANDS, GRID.height, GRID.width), dtype=np.int16)
    stack[ChipBand.VV_POST] = round(post_db * 100)
    stack[ChipBand.VV_PRE] = round(pre_db * 100)
    stack[ChipBand.LABEL] = 1
    d = tmp_path / "events" / "E1"
    d.mkdir(parents=True)
    write_event_stack(d / "stack.tif", stack, GRID)
    (d / "event.json").write_text(json.dumps({"s1_post_datetime": "2024-01-12T05:20:00+00:00", "relative_orbit": 95,
                                              "orbit_state": "descending"}), encoding="utf-8")
    return d


class TestSeasonalReference:
    def test_the_window_is_centred_on_the_same_date_a_year_earlier(self) -> None:
        start, end = anniversary_window(datetime(2024, 1, 12, 5, 20, tzinfo=UTC))
        assert (start.date().isoformat(), end.date().isoformat()) == ("2022-12-13", "2023-02-11")

    def test_too_few_same_orbit_passes_write_a_record_and_no_raster(self, tmp_path) -> None:
        from shapely.geometry import box, mapping

        from floodfusion.data.stac import InMemoryCatalogue

        d = _stack(tmp_path)
        swath = box(14.0, 44.0, 16.0, 45.0)
        items = [{"id": f"s{i}", "collection": "sentinel-1-rtc", "bbox": list(swath.bounds), "geometry": mapping(swath),
                  "properties": {"datetime": ("2023-01-10T05:20:00Z", "2023-01-22T05:20:00Z", "2023-02-03T05:20:00Z")[i], "sat:relative_orbit": orbit,
                                 "sat:orbit_state": "descending"}}
                 for i, orbit in enumerate((95, 95, 44))]
        record = seasonal_reference(d, InMemoryCatalogue.from_dicts(items))
        assert record["status"].startswith("insufficient") and len(record["acquisitions"]) == 2 < MIN_SCENES
        assert not (d / "seasonal.tif").exists() and read_seasonal(d) is None
        assert seasonal_reference(d, None) == record     # cached: no second search

    def test_the_baseline_uses_the_seasonal_reference_and_fails_closed_without_a_record(self, tmp_path) -> None:
        import rasterio

        d = _stack(tmp_path, pre_db=-20.0, post_db=-20.0)      # recent reference already flooded
        cfg = replace(CFG, baseline=replace(CFG.baseline, reference="seasonal", speckle_filter="none"))
        with pytest.raises(FileNotFoundError, match="make seasonal"):
            load_scene(d / "stack.tif", cfg)                     # never built: no silent fallback
        (d / "seasonal.json").write_text(json.dumps({"status": "insufficient: 2 < 3"}), encoding="utf-8")
        scene = load_scene(d / "stack.tif", cfg)
        assert scene.reference == "recent" and np.allclose(scene.vv_pre_db, -20.0)   # counted fallback
        with rasterio.open(d / "seasonal.tif", "w", driver="GTiff", height=GRID.height, width=GRID.width, count=2,
                           dtype="float32", crs=f"EPSG:{GRID.epsg}", transform=GRID.affine) as dst:
            dst.write(np.full((2, GRID.height, GRID.width), -9.0, dtype=np.float32))
        (d / "seasonal.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        scene = load_scene(d / "stack.tif", cfg)
        assert scene.reference == "seasonal" and np.allclose(scene.vv_pre_db, -9.0)
        recent = load_scene(d / "stack.tif", replace(cfg, baseline=replace(cfg.baseline, reference="recent")))
        assert recent.reference == "recent" and np.allclose(recent.vv_pre_db, -20.0)


def test_an_acquisition_without_data_over_the_grid_is_skipped_not_fatal(tmp_path, monkeypatch) -> None:
    from shapely.geometry import box, mapping

    import floodfusion.data.archive as archive
    import floodfusion.data.sources as sources
    from floodfusion.data.stac import InMemoryCatalogue

    d = _stack(tmp_path)
    swath = box(14.0, 44.0, 16.0, 45.0)
    days = ("2022-12-20", "2023-01-01", "2023-01-13", "2023-01-25")
    items = [{"id": f"s{i}", "collection": "sentinel-1-rtc", "bbox": list(swath.bounds), "geometry": mapping(swath),
              "properties": {"datetime": f"{day}T05:20:00Z", "sat:relative_orbit": 95, "sat:orbit_state": "descending"}}
             for i, day in enumerate(days)]

    def read(items, grid, pol):
        if items[0]["id"] == "s1":
            raise ValueError("gamma0_VV: no finite pixels")
        return np.full((grid.height, grid.width), -9.0), {}

    monkeypatch.setattr(sources, "read_s1_rtc", read)
    monkeypatch.setattr(archive, "items_by_id", lambda client, coll, ids: [i for i in items if i["id"] in ids])
    record = seasonal_reference(d, InMemoryCatalogue.from_dicts(items))
    assert record["status"] == "ok" and record["n_scenes"] == 3 and len(record["skipped_acquisitions"]) == 1
    vv, _ = read_seasonal(d)
    assert np.allclose(vv, -9.0)


def _rtc_items(days: tuple[str, ...]) -> list[dict]:
    from shapely.geometry import box, mapping

    swath = box(14.0, 44.0, 16.0, 45.0)
    return [{"id": f"s{i}", "collection": "sentinel-1-rtc", "bbox": list(swath.bounds), "geometry": mapping(swath),
             "properties": {"datetime": f"{day}T05:20:00Z", "sat:relative_orbit": 95, "sat:orbit_state": "descending"}}
            for i, day in enumerate(days)]


def test_too_few_acquisitions_with_data_leave_the_reference_insufficient(tmp_path, monkeypatch) -> None:
    import floodfusion.data.archive as archive
    import floodfusion.data.sources as sources
    from floodfusion.data.stac import InMemoryCatalogue

    d = _stack(tmp_path)
    items = _rtc_items(("2022-12-20", "2023-01-01", "2023-01-13"))

    def read(items, grid, pol):
        if items[0]["id"] != "s0":
            raise ValueError("gamma0_VV: no finite pixels")
        return np.full((grid.height, grid.width), -9.0), {}

    monkeypatch.setattr(sources, "read_s1_rtc", read)
    monkeypatch.setattr(archive, "items_by_id", lambda client, coll, ids: [i for i in items if i["id"] in ids])
    record = seasonal_reference(d, InMemoryCatalogue.from_dicts(items))
    assert record["status"].startswith("insufficient") and record["n_scenes"] == 0
    assert len(record["skipped_acquisitions"]) == 2
    assert json.loads((d / "seasonal.json").read_text(encoding="utf-8"))["status"] == record["status"]


def test_vh_is_optional_when_vv_has_enough_acquisitions(tmp_path, monkeypatch) -> None:
    import floodfusion.data.archive as archive
    import floodfusion.data.sources as sources
    from floodfusion.data.stac import InMemoryCatalogue

    d = _stack(tmp_path)
    items = _rtc_items(("2022-12-20", "2023-01-01", "2023-01-13"))

    def read(items, grid, pol):
        if pol == "VH":
            raise KeyError("VH")   # single-polarisation (1SSV) acquisitions
        return np.full((grid.height, grid.width), -9.0), {}

    monkeypatch.setattr(sources, "read_s1_rtc", read)
    monkeypatch.setattr(archive, "items_by_id", lambda client, coll, ids: [i for i in items if i["id"] in ids])
    record = seasonal_reference(d, InMemoryCatalogue.from_dicts(items))
    assert record["status"] == "ok" and record["n_scenes"] == MIN_SCENES
    vv, vh = read_seasonal(d)
    assert np.allclose(vv, -9.0) and (vh is None or np.isnan(vh).all())


def test_the_cli_counts_each_stack_and_fails_only_when_one_failed(tmp_path, monkeypatch) -> None:
    import floodfusion.data.archive as archive
    import floodfusion.data.seasonal as seasonal

    root = tmp_path / "root" / "events"
    root.mkdir(parents=True)
    for name in ("a", "b"):   # _stack writes <dir>/events/E1; the CLI walks <root>/events/*
        _stack(tmp_path / name).rename(root / name)

    def fake(d, client, overwrite=False):
        if d.name == "b":
            raise RuntimeError("archive unreachable")
        return {"status": "ok"}

    monkeypatch.setattr(archive.IngestSources, "open", classmethod(lambda cls: type("S", (), {"client": None})()))
    monkeypatch.setattr(seasonal, "seasonal_reference", fake)
    assert seasonal.main(["--root", str(tmp_path / "root"), "--workers", "1"]) == 1
    monkeypatch.setattr(seasonal, "seasonal_reference", lambda d, client, overwrite=False: {"status": "ok"})
    assert seasonal.main(["--root", str(tmp_path / "root"), "--workers", "1"]) == 0
