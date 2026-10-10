"""The CEMS benchmark's selection rule: independent products, a close pass, a clean pre-event reference."""

from __future__ import annotations

from datetime import date

import pytest
from shapely.geometry import box, mapping

from floodfusion.cems_benchmark import (
    MIN_COVERAGE,
    acquisitions,
    cached_activations,
    eligible_pairs,
    product_stratum,
    reference_date,
    sample_tiles,
    select_pairs,
    sensor_kind,
)
from floodfusion.data.stac import InMemoryCatalogue

AOI = box(11.0, 44.0, 11.4, 44.3)
SWATH = box(10.5, 43.5, 12.0, 45.0)


def _s1(item_id: str, when: str, orbit: int, state: str = "descending", geom=SWATH) -> dict:
    return {"id": item_id, "collection": "sentinel-1-rtc", "bbox": list(geom.bounds), "geometry": mapping(geom),
            "properties": {"datetime": when, "sat:relative_orbit": orbit, "sat:orbit_state": state}}


def _product(monitoring: int, sensors: list[tuple[str, str]], aoi: int = 1) -> dict:
    return {"type": "DEL", "monitoringNumber": monitoring, "aoiNumber": aoi, "extent": AOI.wkt,
            "downloadPath": f"https://rapidmapping.emergency.copernicus.eu/backend/EMSR999/x{monitoring}.zip",
            "images": [{"sensorName": s, "acquisitionTime": t} for s, t in sensors]}


def _activation(products: list[dict], *, code: str = "EMSR999", event: str = "2024-01-10T00:00:00",
                extent=AOI) -> dict:
    return {"code": code, "category": "Flood", "eventTime": event, "extent": extent.wkt,
            "aois": [{"number": 1, "name": "A", "extent": AOI.wkt, "products": products}]}


class TestSensors:
    @pytest.mark.parametrize(("name", "kind"), [
        ("COSMO-SkyMed Second Generation", "sar"), ("PAZ", "sar"), ("RADARSAT-2", "sar"), ("ICEYE", "sar"),
        ("Pleiades Neo", "optical"), ("WorldView-3", "optical"), ("SPOT-6-7", "optical"),
        ("Sentinel-1", "sentinel"), ("Sentinel-2B", "sentinel"), ("", "unknown"), ("Mystery-1", "unknown"),
    ])
    def test_sensor_kinds(self, name, kind) -> None:
        assert sensor_kind(name) == kind

    def test_a_product_touching_sentinel_or_an_unknown_sensor_is_not_independent(self) -> None:
        assert product_stratum(["COSMO-SkyMed", "PAZ"]) == "sar"
        assert product_stratum(["Pleiades"]) == "optical"
        assert product_stratum(["COSMO-SkyMed", "Pleiades"]) == "mixed"
        assert product_stratum(["COSMO-SkyMed", "Sentinel-1"]) is None
        assert product_stratum(["Mystery-1"]) is None
        assert product_stratum([]) is None


class TestReferenceDate:
    def test_a_week_before_the_event(self) -> None:
        assert reference_date(_activation([]), []) == date(2024, 1, 3)

    def test_an_earlier_overlapping_flood_moves_it_back(self) -> None:
        earlier = _activation([], code="EMSR998", event="2023-12-20T00:00:00")
        far = _activation([], code="EMSR997", event="2023-12-25T00:00:00", extent=box(30.0, 10.0, 31.0, 11.0))
        old = _activation([], code="EMSR996", event="2023-09-01T00:00:00")
        assert reference_date(_activation([]), [earlier, far, old]) == date(2023, 12, 13)


class TestEligibility:
    def test_frames_of_one_acquisition_merge(self) -> None:
        north, south = box(10.5, 44.15, 12.0, 45.0), box(10.5, 43.5, 12.0, 44.15)
        cat = InMemoryCatalogue.from_dicts([_s1("f1", "2024-01-12T05:20:00Z", 95, geom=north),
                                            _s1("f2", "2024-01-12T05:20:25Z", 95, geom=south)])
        from datetime import UTC, datetime
        acq = acquisitions(cat, AOI, datetime(2024, 1, 12, tzinfo=UTC), datetime(2024, 1, 13, tzinfo=UTC))
        assert len(acq) == 1 and acq[0].frame_ids == ("f1", "f2")
        assert acq[0].footprint.contains(AOI)

    def test_the_closest_covering_pass_with_a_same_orbit_reference_is_kept(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post_far", "2024-01-12T22:00:00Z", 95),            # +17 h
            _s1("post_near", "2024-01-12T07:00:00Z", 44, "ascending"),  # +2 h
            _s1("pre_95", "2023-12-20T05:20:00Z", 95),
            _s1("pre_44", "2023-12-15T17:00:00Z", 44, "ascending"),
        ])
        act = _activation([_product(0, [("COSMO-SkyMed", "2024-01-12T05:00:00")])])
        [pair] = eligible_pairs(act, cat)
        assert (pair.relative_orbit, pair.orbit_state, pair.offset_h) == (44, "ascending", 2.0)
        assert pair.stratum == "sar" and pair.coverage == pytest.approx(1.0)
        assert pair.reference_before == "2024-01-03" and pair.n_reference_scenes == 1
        assert pair.product == "EMSR999_AOI01_DEL_PRODUCT"

    def test_sentinel_products_far_passes_and_partial_coverage_are_ineligible(self) -> None:
        sliver = box(11.0, 44.0, 11.12, 44.3)   # 30 % of the AOI's width
        assert 0.3 < MIN_COVERAGE
        cat = InMemoryCatalogue.from_dicts([
            _s1("partial", "2024-01-12T06:00:00Z", 95, geom=sliver),
            _s1("late", "2024-01-14T06:00:00Z", 95),                 # +49 h
            _s1("pre", "2023-12-20T05:20:00Z", 95),
        ])
        sentinel = _product(0, [("Sentinel-1", "2024-01-12T05:00:00")])
        independent = _product(1, [("PAZ", "2024-01-12T05:00:00")])
        assert eligible_pairs(_activation([sentinel, independent]), cat) == []

    def test_no_same_orbit_reference_before_the_cutoff_means_ineligible(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post", "2024-01-12T06:00:00Z", 95),
            _s1("pre_too_late", "2024-01-05T06:00:00Z", 95),          # after the reference date (Jan 3)
            _s1("pre_other_orbit", "2023-12-20T17:00:00Z", 44, "ascending"),
        ])
        assert eligible_pairs(_activation([_product(0, [("RADARSAT-2", "2024-01-12T05:00:00")])]), cat) == []

    def test_a_product_without_a_published_package_is_ineligible(self) -> None:
        cat = InMemoryCatalogue.from_dicts([_s1("post", "2024-01-12T06:00:00Z", 95), _s1("pre", "2023-12-20T05:20:00Z", 95)])
        unpublished = {**_product(0, [("PAZ", "2024-01-12T05:00:00")]), "downloadPath": ""}
        assert eligible_pairs(_activation([unpublished]), cat) == []
        assert len(eligible_pairs(_activation([_product(0, [("PAZ", "2024-01-12T05:00:00")])]), cat)) == 1

    def test_non_flood_activations_are_skipped(self) -> None:
        act = {**_activation([_product(0, [("PAZ", "2024-01-12T05:00:00")])]), "category": "Wildfire"}
        assert eligible_pairs(act, InMemoryCatalogue.from_dicts([])) == []


class TestActivationCache:
    def test_records_are_fetched_once_and_unknown_codes_remembered(self, tmp_path) -> None:
        calls: list[str] = []

        def fetch(code: str) -> dict:
            calls.append(code)
            if code == "EMSR102":
                raise LookupError(code)
            return {"code": code}

        pauses: list[float] = []
        codes = ["EMSR101", "EMSR102", "EMSR103"]
        first = cached_activations(codes, tmp_path, fetch=fetch, sleep=pauses.append)
        again = cached_activations(codes, tmp_path, fetch=fetch, sleep=pauses.append)
        assert [r["code"] for r in first] == [r["code"] for r in again] == ["EMSR101", "EMSR103"]
        assert calls == codes and len(pauses) == 3

    def test_a_refusal_is_asked_again_then_recorded_and_other_failures_stop_the_sweep(self, tmp_path) -> None:
        from floodfusion.data.http import HTTPStatusError

        calls: list[str] = []

        def fetch(code: str) -> dict:
            calls.append(code)
            if code == "EMSR102":
                raise HTTPStatusError("HTTP 403 for https://rapidmapping.emergency.copernicus.eu/...", 403)
            if code == "EMSR104":
                raise HTTPStatusError("HTTP 404 for https://rapidmapping.emergency.copernicus.eu/...", 404)
            return {"code": code}

        with pytest.raises(HTTPStatusError, match="404"):
            cached_activations(["EMSR101", "EMSR102", "EMSR103", "EMSR104"], tmp_path, fetch=fetch, sleep=lambda s: None)
        assert calls.count("EMSR102") == 3                   # asked three times before it counts as not public
        assert (tmp_path / "EMSR101.json").is_file() and (tmp_path / "EMSR102.restricted").is_file()
        assert (tmp_path / "EMSR103.json").is_file() and not (tmp_path / "EMSR104.missing").exists()
        from floodfusion.cems_benchmark import cache_census

        census = cache_census(["EMSR101", "EMSR102", "EMSR103", "EMSR104"], tmp_path)
        assert census == {"public": ["EMSR101", "EMSR103"], "missing": [], "restricted": ["EMSR102"]}

    def test_codes_are_validated_before_they_name_a_file(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="activation code"):
            cached_activations(["../outside"], tmp_path, fetch=lambda c: pytest.fail("fetched"), sleep=lambda s: None)
        from floodfusion.cems_benchmark import pair_key

        with pytest.raises(ValueError, match="activation code"):
            pair_key({"activation": "EMSR1/../../x", "aoi": 1})


class TestRateLimit:
    def test_a_rate_limited_search_is_retried_with_backoff(self) -> None:
        from datetime import UTC, datetime

        class Flaky:
            calls = 0

            def search(self, **kwargs):
                Flaky.calls += 1
                if Flaky.calls < 3:
                    raise RuntimeError("You have exceeded a rate limit. Contact planetarycomputer@microsoft.com")
                return InMemoryCatalogue.from_dicts([_s1("a", "2024-01-12T05:20:00Z", 95)]).search(**kwargs)

        waits: list[float] = []
        acq = acquisitions(Flaky(), AOI, datetime(2024, 1, 12, tzinfo=UTC), datetime(2024, 1, 13, tzinfo=UTC),
                           sleep=waits.append)
        assert len(acq) == 1 and waits == [30.0, 60.0]

    def test_other_errors_are_not_retried(self) -> None:
        from datetime import UTC, datetime

        class Broken:
            def search(self, **kwargs):
                raise RuntimeError("HTTP 500")

        with pytest.raises(RuntimeError, match="500"):
            acquisitions(Broken(), AOI, datetime(2024, 1, 12, tzinfo=UTC), datetime(2024, 1, 13, tzinfo=UTC),
                         sleep=lambda s: pytest.fail("retried"))


class TestCaps:
    def test_at_most_four_aois_per_activation_closest_in_time_first(self) -> None:
        pairs = [{"activation": "EMSR1", "aoi": i, "offset_h": off} for i, off in
                 enumerate([5.0, -1.0, 12.0, 0.5, -1.0, 23.0], start=1)] + [{"activation": "EMSR2", "aoi": 1, "offset_h": 3.0}]
        kept = select_pairs(pairs)
        assert [(p["activation"], p["aoi"]) for p in kept] == [("EMSR1", 4), ("EMSR1", 2), ("EMSR1", 5), ("EMSR1", 1),
                                                                ("EMSR2", 1)]

    def test_tiles_are_sampled_blind_and_reproducibly(self) -> None:
        tiles = [(f"t{i:02d}", None) for i in range(40)]
        first = sample_tiles(tiles, "EMSR1_AOI01")
        assert len(first) == 16 and first == sample_tiles(tiles, "EMSR1_AOI01")
        assert [t for t, _ in first] == sorted(t for t, _ in first)
        assert first != sample_tiles(tiles, "EMSR1_AOI02")
        assert sample_tiles(tiles[:5], "EMSR1_AOI01") == tiles[:5]
