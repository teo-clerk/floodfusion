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

    def test_the_baseline_uses_the_seasonal_reference_where_it_exists(self, tmp_path) -> None:
        import rasterio

        d = _stack(tmp_path, pre_db=-20.0, post_db=-20.0)      # recent reference already flooded
        cfg = replace(CFG, baseline=replace(CFG.baseline, reference="seasonal", speckle_filter="none"))
        assert np.allclose(load_scene(d / "stack.tif", cfg).vv_pre_db, -20.0)   # no seasonal.tif: recent kept
        with rasterio.open(d / "seasonal.tif", "w", driver="GTiff", height=GRID.height, width=GRID.width, count=2,
                           dtype="float32", crs=f"EPSG:{GRID.epsg}", transform=GRID.affine) as dst:
            dst.write(np.full((2, GRID.height, GRID.width), -9.0, dtype=np.float32))
        assert np.allclose(load_scene(d / "stack.tif", cfg).vv_pre_db, -9.0)
        assert np.allclose(load_scene(d / "stack.tif", replace(cfg, baseline=replace(cfg.baseline, reference="recent")))
                           .vv_pre_db, -20.0)
