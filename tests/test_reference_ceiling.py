"""The reference ceiling (``scripts/reference_ceiling.py``; protocol v1.3 §2.3 and §5), offline."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

_spec = importlib.util.spec_from_file_location("reference_ceiling",
                                               Path(__file__).parents[1] / "scripts" / "reference_ceiling.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rc          # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(rc)

_URL = "https://rapidmapping.emergency.copernicus.eu/backend/EMSR700/x.zip"


def _product(monitoring: int, sensor: str, when: str, *, kind: str = "DEL") -> dict:
    return {"type": kind, "monitoringNumber": monitoring, "downloadPath": _URL,
            "images": [{"sensorName": sensor, "acquisitionTime": when}]}


def _record(code: str = "EMSR700", category: str = "Flood", products: list | None = None) -> dict:
    products = products if products is not None else [
        _product(0, "COSMO-SkyMed", "2023-05-01T05:00:00"),
        _product(1, "Pleiades", "2023-05-01T17:00:00"),          # 12 h later: a candidate
        _product(2, "ICEYE", "2023-05-03T05:00:00"),             # 36 h later: too far
        _product(3, "PAZ", "2023-05-03T06:00:00", kind="GRA"),   # not a delineation
    ]
    return {"code": code, "category": category, "aois": [{"number": 1, "name": "x", "products": products}]}


def test_candidates_are_consecutive_delineations_at_most_24_h_apart():
    [c] = rc.candidates([_record()])
    assert (c.activation, c.aoi, c.first.monitoring, c.second.monitoring) == ("EMSR700", 1, 0, 1)
    assert c.gap_h == pytest.approx(12.0)
    assert c.sensor_pair == "radar–optical" and not c.shared_image
    assert c.key == "EMSR700_AOI01_PRODUCT_MONIT01"


def test_candidates_order_products_by_their_latest_image_not_by_monitoring_number():
    record = _record(products=[_product(0, "PAZ", "2023-05-01T17:00:00"), _product(1, "PAZ", "2023-05-01T05:00:00")])
    [c] = rc.candidates([record])
    assert (c.first.monitoring, c.second.monitoring) == (1, 0)


def test_a_shared_source_image_is_marked():
    record = _record(products=[_product(0, "PAZ", "2023-05-01T05:00:00"), _product(1, "PAZ", "2023-05-01T05:00:00")])
    [c] = rc.candidates([record])
    assert c.shared_image and c.gap_h == 0.0 and c.sensor_pair == "radar–radar"


def test_candidates_skip_other_categories_and_refuse_the_blind_set():
    assert rc.candidates([_record(category="Wildfire")]) == []
    with pytest.raises(ValueError, match="EMSR899"):
        rc.candidates([_record(code="EMSR899")])


@pytest.mark.parametrize(("sensor", "kind"), [("Sentinel-1A", "radar"), ("Sentinel-2B", "optical"),
                                              ("COSMO-SkyMed", "radar"), ("Pleiades", "optical"),
                                              ("Mystery-Sat", "unknown"), ("", "unknown")])
def test_image_kind(sensor: str, kind: str):
    assert rc.image_kind(sensor) == kind


def test_sensor_pair_is_canonical_and_mixed_products_are_other():
    assert rc.sensor_pair("optical", "radar") == rc.sensor_pair("radar", "optical") == "radar–optical"
    assert rc.sensor_pair("mixed", "radar") == "other"


def test_counts_compare_two_maps_on_their_common_observed_region():
    # Regions overlap on lon 0.005–0.010; inside it, map A floods 0.005–0.008 and map B 0.006–0.009.
    counts = rc.counts(box(0.0, 0.0, 0.010, 0.01), box(0.005, 0.0, 0.015, 0.01),
                       [box(0.005, 0.0, 0.008, 0.01)], [box(0.006, 0.0, 0.009, 0.01)])
    assert counts["region_px"] > 0
    assert rc.iou(counts) == pytest.approx(0.5, abs=0.03)          # both 0.006–0.008 over union 0.005–0.009


def test_counts_are_none_without_a_common_region():
    assert rc.counts(box(0, 0, 0.01, 0.01), box(0.02, 0, 0.03, 0.01), [], []) is None


def test_summary_pools_pairs_per_activation_and_averages_activations():
    pairs = [
        {"activation": "EMSR700", "gap_h": 3.0, "sensor_pair": "radar–radar", "shared_image": False,
         "counts": {"region_px": 100, "both": 10, "only_first": 10, "only_second": 0}},
        {"activation": "EMSR700", "gap_h": 12.0, "sensor_pair": "radar–optical", "shared_image": False,
         "counts": {"region_px": 100, "both": 10, "only_first": 0, "only_second": 0}},
        {"activation": "EMSR701", "gap_h": 12.0, "sensor_pair": "radar–radar", "shared_image": False,
         "counts": {"region_px": 100, "both": 0, "only_first": 5, "only_second": 5}},
        {"activation": "EMSR702", "gap_h": 12.0, "sensor_pair": "radar–radar", "shared_image": False,
         "counts": {"region_px": 100, "both": 0, "only_first": 0, "only_second": 0}},   # no flood in either: no IoU
    ]
    summary = rc.summarise(pairs)
    assert summary["all"]["activations"] == 2 and summary["all"]["undefined_activations"] == 1
    assert summary["all"]["macro_iou"] == pytest.approx((20 / 30 + 0.0) / 2)
    assert summary["gap"]["< 6 h"]["macro_iou"] == pytest.approx(0.5)
    assert summary["sensor_pair"]["radar–optical"]["macro_iou"] == pytest.approx(1.0)


def test_iou_is_nan_when_neither_map_floods():
    assert math.isnan(rc.iou({"region_px": 5, "both": 0, "only_first": 0, "only_second": 0}))


def test_carried_over_share_measures_flood_drawn_before_the_first_products_image():
    from datetime import UTC, datetime

    t0 = datetime(2023, 5, 1, 5, tzinfo=UTC)
    flood = [(box(0, 0, 1, 1), t0), (box(1, 0, 4, 1), datetime(2023, 5, 1, 17, tzinfo=UTC)), (box(4, 0, 5, 1), None)]
    assert rc.carried_over_share(flood, first_latest=t0) == pytest.approx(1 / 5)
