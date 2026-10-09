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
    assert [t for t, _ in tiles] == ["t00", "t01", "t10", "t11"]
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


def _product_dir(tmp_path: Path) -> Path:
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    d = tmp_path / "EMSR999_AOI01_DEL_PRODUCT"
    d.mkdir()
    gpd.GeoDataFrame({"emsr_id": ["EMSR999"]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs=4326).to_file(
        d / "EMSR999_AOI01_DEL_PRODUCT_areaOfInterestA_v1.json", driver="GeoJSON")
    gpd.GeoDataFrame({"obj_type": ["Image Footprint", "Image Footprint", "Not Analysed - No data"],
                      "or_src_id": [1, 2, 2]},
                     geometry=[box(-5, -5, 15, 15), box(0, 0, 6, 10), box(0, 0, 1, 1)], crs=4326).to_file(
        d / "EMSR999_AOI01_DEL_PRODUCT_imageFootprintA_v1.json", driver="GeoJSON")
    pyogrio.write_dataframe(pd.DataFrame({"src_date": ["01/01/2020", "17/05/2023"], "source_tm": ["T10:00Z", "T16:56Z"],
                                          "eventphase": ["Pre-event", "Post-event"], "src_id": [1, 2]}),
                            d / "EMSR999_AOI01_DEL_PRODUCT_source_v1.dbf", driver="ESRI Shapefile")
    return d


def test_scored_region_is_the_post_event_footprint_inside_the_aoi_minus_not_analysed(tmp_path) -> None:
    region = scored_region(_product_dir(tmp_path))
    # Post-event image (src 2) covers x in [0, 6] of the AOI; the pre-event image (src 1) does not count.
    assert region.bounds == pytest.approx((0.0, 0.0, 6.0, 10.0))
    assert region.area == pytest.approx(60.0 - 1.0)


def test_score_counts_only_scored_pixels() -> None:
    pred = np.array([[True, True], [False, False]])
    label = np.array([[True, False], [True, False]])
    scored = np.array([[True, True], [True, False]])
    assert _score(pred, label, scored) == ConfusionMatrix(1, 1, 1, 0)


def test_report_marks_the_nearest_product_and_orders_the_recession(tmp_path) -> None:
    from floodfusion.case_study import _Model

    case = {"activation": "EMSR999", "aoi": 1, "reference_before": "2023-05-01",
            "products": [{"tag": "EMSR999_AOI01_DEL_PRODUCT", "sensor": "CSK", "time": "2023-05-17T16:56:00+00:00",
                          "flood_km2_in_region": 80.0},
                         {"tag": "EMSR999_AOI01_DEL_MONIT01", "sensor": "RS2", "time": "2023-05-18T16:55:00+00:00",
                          "flood_km2_in_region": 100.0}],
            "passes": [{"tag": "S1_20230518T1658_o044", "when": "2023-05-18T16:58:50+00:00",
                        "label_product": "EMSR999_AOI01_DEL_MONIT01", "tiles": {}}]}
    model = _Model("fusion", None, 0.25)
    cms = {("S1_20230518T1658_o044", "EMSR999_AOI01_DEL_PRODUCT", "fusion"): ConfusionMatrix(50, 30, 20, 900),
           ("S1_20230518T1658_o044", "EMSR999_AOI01_DEL_MONIT01", "fusion"): ConfusionMatrix(70, 10, 10, 900)}
    area = {"common": 400.0, "cems:EMSR999_AOI01_DEL_PRODUCT": 70.0, "cems:EMSR999_AOI01_DEL_MONIT01": 75.0,
            "fusion:S1_20230518T1658_o044": 72.0}
    result = _report(case, [model], cms, area, {"S1_20230518T1658_o044": [False, True]}, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    assert "**0.778 (+0 h)**" in text and "0.500 (+24 h)" in text
    assert result["iou"]["fusion"]["S1_20230518T1658_o044"]["EMSR999_AOI01_DEL_MONIT01"] == pytest.approx(0.7778, abs=1e-4)
    rows = [line for line in text.splitlines() if line.startswith("| CEMS") or line.startswith("| fusion,")]
    assert rows[0].startswith("| CEMS DEL_PRODUCT") and rows[-1].startswith("| fusion, S1_20230518T1658_o044")
    assert json.loads((tmp_path / "r.json").read_text())["area_km2"]["fusion, S1_20230518T1658_o044"] == 72.0
