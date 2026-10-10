"""The CEMS case study: tiling, pass discovery, the region a product observed, the report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box, mapping

from floodfusion.case_study import _report, _score, find_passes, scored_region, tiles_for
from floodfusion.data.stac import InMemoryCatalogue
from floodfusion.metrics import ConfusionMatrix


def test_tiles_cover_the_bounds_without_overhang() -> None:
    tiles = tiles_for((11.7, 44.1, 12.1, 44.4), (0.2, 0.15))
    assert [t for t, _ in tiles] == ["t00_00", "t00_01", "t01_00", "t01_01"]
    assert tiles[0][1] == pytest.approx((11.7, 44.25, 11.9, 44.4))
    assert tiles[-1][1] == pytest.approx((11.9, 44.1, 12.1, 44.25))
    ragged = tiles_for((0.0, 0.0, 0.3, 0.2), (0.2, 0.15))
    assert ragged[-1][1] == pytest.approx((0.2, 0.0, 0.3, 0.05))


SWATH = box(11.0, 43.5, 13.0, 45.5)


def _s1(item_id: str, when: str, orbit: int, state: str, geom=SWATH) -> dict:
    return {"id": item_id, "collection": "sentinel-1-rtc", "bbox": list(geom.bounds), "geometry": mapping(geom),
            "properties": {"datetime": when, "sat:relative_orbit": orbit, "sat:orbit_state": state}}


def test_frames_of_one_pass_merge_and_distant_footprints_are_ignored() -> None:
    aoi = box(11.6, 44.0, 12.5, 44.6)
    catalogue = InMemoryCatalogue.from_dicts([
        _s1("a1", "2023-05-18T16:58:37Z", 44, "ascending"), _s1("a2", "2023-05-18T16:59:02Z", 44, "ascending"),
        _s1("d1", "2023-05-22T05:19:46Z", 95, "descending"),
        _s1("far", "2023-05-20T05:00:00Z", 22, "descending", geom=box(20.0, 40.0, 21.0, 41.0)),
        _s1("old", "2023-05-10T05:19:46Z", 95, "descending"),
    ])
    passes = find_passes(catalogue, aoi, date(2023, 5, 16), date(2023, 5, 23))
    assert [(p.relative_orbit, p.orbit_state) for p in passes] == [(44, "ascending"), (95, "descending")]
    assert passes[0].tag == "S1_20230518T1658_o044"


def test_tile_ids_never_collide() -> None:
    ids = [t for t, _ in tiles_for((0.0, 0.0, 2.4, 2.4), (0.2, 0.2))]   # 12 x 12 tiles
    assert len(ids) == len(set(ids)) == 144


def _product_dir(tmp_path: Path, flood_src: int | None = 2) -> Path:
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    d = tmp_path / "EMSR999_AOI01_DEL_PRODUCT"
    d.mkdir()
    flood = [box(1.0, 1.0, 2.0, 2.0)] if flood_src is not None else []
    gpd.GeoDataFrame({"event_type": ["5-Flood"] * len(flood), "dmg_src_id": [flood_src] * len(flood)},
                     geometry=flood, crs=4326).to_file(d / "EMSR999_AOI01_DEL_PRODUCT_observedEventA_v1.json",
                                                       driver="GeoJSON")
    gpd.GeoDataFrame({"emsr_id": ["EMSR999"]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs=4326).to_file(
        d / "EMSR999_AOI01_DEL_PRODUCT_areaOfInterestA_v1.json", driver="GeoJSON")
    gpd.GeoDataFrame({"obj_type": ["Image Footprint", "Image Footprint", "Not Analysed - No data"],
                      "or_src_id": [1, 2, 2]},
                     geometry=[box(-5, -5, 15, 15), box(0, 0, 6, 10), box(0, 0, 1, 1)], crs=4326).to_file(
        d / "EMSR999_AOI01_DEL_PRODUCT_imageFootprintA_v1.json", driver="GeoJSON")
    pyogrio.write_dataframe(pd.DataFrame({"src_date": ["01/01/2020", "17/05/2023", "16/05/2023"],
                                          "source_tm": ["T10:00Z", "T16:56Z", "T05:00Z"],
                                          "eventphase": ["Pre-event", "Post-event", "Post-event"], "src_id": [1, 2, 3]}),
                            d / "EMSR999_AOI01_DEL_PRODUCT_source_v1.dbf", driver="ESRI Shapefile")
    return d


def test_scored_region_is_the_footprint_of_the_images_the_polygons_came_from(tmp_path) -> None:
    region = scored_region(_product_dir(tmp_path))
    # The polygons come from src 2, which covers x in [0, 6]; the pre-event image (src 1) does not count.
    assert region.bounds == pytest.approx((0.0, 0.0, 6.0, 10.0))
    assert region.area == pytest.approx(60.0 - 1.0)


def test_a_monitoring_product_is_scored_only_where_its_new_image_looked(tmp_path) -> None:
    """Review finding: an earlier image's footprint (src 3, listed in the layer) was scored as 'no flood'."""
    import geopandas as gpd

    d = _product_dir(tmp_path)
    frame = gpd.read_file(d / "EMSR999_AOI01_DEL_PRODUCT_imageFootprintA_v1.json")
    extra = gpd.GeoDataFrame({"obj_type": ["Image Footprint"], "or_src_id": [3]}, geometry=[box(5, 0, 10, 10)], crs=4326)
    gpd.pd.concat([frame, extra]).to_file(d / "EMSR999_AOI01_DEL_PRODUCT_imageFootprintA_v1.json", driver="GeoJSON")
    assert scored_region(d).bounds == pytest.approx((0.0, 0.0, 6.0, 10.0))   # src 3 adds nothing


def test_no_matching_footprint_raises_instead_of_scoring_the_whole_aoi(tmp_path) -> None:
    d = _product_dir(tmp_path, flood_src=7)
    with pytest.raises(ValueError, match="no image footprint"):
        scored_region(d)


def test_score_counts_only_scored_pixels() -> None:
    pred = np.array([[True, True], [False, False]])
    label = np.array([[True, False], [True, False]])
    scored = np.array([[True, True], [True, False]])
    assert _score(pred, label, scored) == ConfusionMatrix(1, 1, 1, 0)


def test_report_marks_the_nearest_product_and_orders_the_recession(tmp_path) -> None:
    from floodfusion.case_study import _Model, _Tally

    case = {"activation": "EMSR999", "aoi": 1, "reference_before": "2023-05-01",
            "products": [{"tag": "EMSR999_AOI01_DEL_PRODUCT", "sensor": "CSK|x", "time": "2023-05-17T16:56:00+00:00",
                          "flood_km2": 80.0, "region_km2": 900.0},
                         {"tag": "EMSR999_AOI01_DEL_MONIT01", "sensor": "RS2", "time": "2023-05-18T16:55:00+00:00",
                          "flood_km2": 100.0, "region_km2": 800.0}],
            "passes": [{"tag": "S1_20230518T1658_o044", "when": "2023-05-18T16:58:50+00:00",
                        "label_product": "EMSR999_AOI01_DEL_MONIT01", "tiles": {}}]}
    model = _Model("fusion", None, 0.25)
    near, far = ConfusionMatrix(70, 10, 10, 900), ConfusionMatrix(50, 30, 20, 900)
    oracle = ConfusionMatrix(75, 8, 7, 900)
    tally = _Tally(common={("S1_20230518T1658_o044", "EMSR999_AOI01_DEL_PRODUCT", "fusion"): [far, far],
                           ("S1_20230518T1658_o044", "EMSR999_AOI01_DEL_MONIT01", "fusion"): [near, oracle]},
                   own={("S1_20230518T1658_o044", "EMSR999_AOI01_DEL_MONIT01", "fusion"): [near, oracle]},
                   area={"common": 400.0, "cems:EMSR999_AOI01_DEL_PRODUCT": 70.0,
                         "cems:EMSR999_AOI01_DEL_MONIT01": 75.0, "fusion:S1_20230518T1658_o044": 72.0},
                   optical={"S1_20230518T1658_o044": [False, True]})
    result = _report(case, [model], tally, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    assert "**0.778 / 0.833 (+0 h)**" in text and "0.500 / 0.500 (+24 h)" in text
    assert result["common"]["fusion"]["S1_20230518T1658_o044"]["EMSR999_AOI01_DEL_MONIT01"]["iou_tau"] == pytest.approx(0.7778, abs=1e-4)
    assert "CSK/x" in text and "CSK|x" not in text                          # API strings cannot break the table
    rows = [line for line in text.splitlines() if line.startswith("| CEMS") or line.startswith("| fusion,")]
    assert rows[0].startswith("| CEMS DEL_PRODUCT") and rows[-1].startswith("| fusion, S1_20230518T1658_o044")
    assert json.loads((tmp_path / "r.json").read_text())["area_km2"]["fusion, S1_20230518T1658_o044"] == 72.0


def _baseline_stack(tmp_path: Path, *, vv_left: float, vv_right: float, label: int, no_sar: bool = False) -> Path:
    from floodfusion.data.chipping import AnalysisGrid, write_event_stack
    from floodfusion.data.encoding import N_CHIP_BANDS, ChipBand

    rng = np.random.default_rng(0)
    stack = np.zeros((N_CHIP_BANDS, 64, 64), dtype=np.int16)
    vv = np.where(np.arange(64)[None, :] < 32, vv_left, vv_right) + rng.normal(0, 0.5, (64, 64))
    dry = np.full((64, 64), vv_right) + rng.normal(0, 0.5, (64, 64))      # before the flood: land everywhere
    stack[ChipBand.VV_POST], stack[ChipBand.VV_PRE] = np.round(vv * 100), np.round(dry * 100)
    stack[ChipBand.VH_POST], stack[ChipBand.VH_PRE] = np.round((vv - 6) * 100), np.round((dry - 6) * 100)
    stack[ChipBand.LABEL] = label
    if no_sar:  # the int16 nodata sentinel, exactly: no SAR coverage on this tile
        stack[[ChipBand.VV_POST, ChipBand.VH_POST, ChipBand.VV_PRE, ChipBand.VH_PRE, ChipBand.DELTA_VV]] = -32768
    path = tmp_path / "stack.tif"
    write_event_stack(path, stack, AnalysisGrid(epsg=32632, transform=(10.0, 0.0, 7e5, 0.0, -10.0, 4.9e6),
                                               width=64, height=64))
    return path


def test_the_baseline_maps_a_case_tile_that_carries_no_label(tmp_path) -> None:
    """Release finding: the case-study baseline chose its pixels by label and crashed on unlabelled tiles."""
    from floodfusion.case_study import _baseline

    # Arrange: the left half darkened since the dry reference; no label anywhere (nodata).
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval_test.json").write_text(json.dumps({"operating_offset_db": 0.0}))
    stack = _baseline_stack(tmp_path, vv_left=-22.0, vv_right=-8.0, label=-32768)
    model = _baseline(run, Path(__file__).resolve().parents[1] / "configs", [])

    # Act
    flood = model.predict(stack) > 0.5

    # Assert
    assert flood[:, :28].mean() > 0.9 and flood[:, 36:].mean() < 0.1


def test_a_case_tile_without_usable_sar_maps_no_flood(tmp_path) -> None:
    from floodfusion.case_study import _baseline

    run = tmp_path / "run"
    run.mkdir()
    (run / "eval_test.json").write_text(json.dumps({"operating_offset_db": 0.0}))
    stack = _baseline_stack(tmp_path, vv_left=-22.0, vv_right=-8.0, label=-32768, no_sar=True)
    assert not model_predicts_flood(_baseline(run, Path(__file__).resolve().parents[1] / "configs", []), stack)


def model_predicts_flood(model, stack) -> bool:
    return bool((model.predict(stack) > 0.5).any())
