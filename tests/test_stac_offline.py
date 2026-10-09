"""Offline behaviour of the STAC matcher beyond the given ``test_stac_query.py``."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from floodfusion.data.stac import (
    EventQuery,
    InMemoryCatalogue,
    MatchedScenes,
    load_matched_scenes,
    profile_for,
    save_matched_scenes,
    search_matched_pairs,
    summarise_optical_availability,
)

BBOX = [9.8, 44.9, 10.7, 45.7]


def _item(item_id: str, collection: str, when: str, **props) -> dict:
    return {"id": item_id, "collection": collection, "bbox": props.pop("bbox", BBOX),
            "properties": {"datetime": when, **props}, "assets": {}}


def _s1(item_id: str, when: str, orbit: int = 117, state: str = "ascending", **extra) -> dict:
    props = {"sat:relative_orbit": orbit, "sat:orbit_state": state, "sar:instrument_mode": "IW",
             "sar:polarizations": ["VV", "VH"], **extra}
    return _item(item_id, "sentinel-1-rtc", when, **props)


def _query(**kw) -> EventQuery:
    return EventQuery("e0", (10.0, 45.0, 10.5, 45.5), datetime(2023, 11, 4), datetime(2023, 11, 7), **kw)


PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


class TestProfiles:
    def test_host_selects_profile(self) -> None:
        assert profile_for(PC).s1_collection == "sentinel-1-rtc"
        # Exact https host, never a substring (security review): these are NOT Planetary Computer.
        for spoof in ("https://planetarycomputer.evil.example/api", "https://planetarycomputer.microsoft.com@evil.example/",
                      "http://planetarycomputer.microsoft.com/api/stac/v1"):
            assert profile_for(spoof).name == "generic"
        assert profile_for("https://catalogue.dataspace.copernicus.eu/stac").dem_collection == "cop-dem-glo-30-dged-cog"
        assert profile_for("https://example.org/stac").name == "generic"


class TestMatching:
    def test_flood_window_is_day_inclusive(self) -> None:
        """flood_end = 2023-11-07 00:00 must include the 17:07 pass of that day."""
        cat = InMemoryCatalogue.from_dicts([
            _s1("post", "2023-11-07T17:07:13Z"), _s1("pre", "2023-10-14T17:07:13Z"),
        ])
        assert search_matched_pairs(_query(), PC, client=cat).s1_post_id == "post"

    def test_earliest_post_and_latest_same_orbit_pre(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post_late", "2023-11-06T17:07:13Z"), _s1("post_early", "2023-11-05T05:20:00Z", 168, "descending"),
            _s1("pre_168_old", "2023-10-01T05:20:00Z", 168, "descending"),
            _s1("pre_168_new", "2023-10-24T05:20:00Z", 168, "descending"),
            _s1("pre_117", "2023-10-31T17:07:13Z"),
        ])
        m = search_matched_pairs(_query(), PC, client=cat)
        assert (m.s1_post_id, m.s1_pre_id, m.relative_orbit, m.orbit_state) == (
            "post_early", "pre_168_new", 168, "descending")

    def test_frames_of_one_acquisition_are_grouped(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post_a", "2023-11-07T17:07:13Z", bbox=[9.8, 44.0, 10.7, 45.1]),
            _s1("post_b", "2023-11-07T17:07:38Z", bbox=[9.8, 45.0, 10.7, 46.0]),
            _s1("pre_a", "2023-10-14T17:07:13Z"), _s1("pre_b", "2023-10-14T17:07:38Z"),
        ])
        m = search_matched_pairs(_query(), PC, client=cat)
        assert m.s1_post_frames == ("post_a", "post_b") and m.s1_pre_frames == ("pre_a", "pre_b")
        assert m.s1_post_id == "post_b"  # larger overlap with the AOI

    def test_unusable_modes_and_polarisations_are_ignored(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("ew", "2023-11-05T17:07:13Z", **{"sar:instrument_mode": "EW"}),
            _s1("single_pol", "2023-11-05T18:07:13Z", **{"sar:polarizations": ["VV"]}),
            _s1("pre", "2023-10-14T17:07:13Z"),
        ])
        with pytest.raises(LookupError, match="no Sentinel-1"):
            search_matched_pairs(_query(), PC, client=cat)

    def test_pre_window_is_bounded(self) -> None:
        cat = InMemoryCatalogue.from_dicts([_s1("post", "2023-11-05T17:07:13Z"), _s1("pre", "2023-08-01T17:07:13Z")])
        with pytest.raises(LookupError, match="relative orbit 117"):
            search_matched_pairs(_query(pre_event_window_days=30), PC, client=cat)

    def test_s2_and_dem_are_reported(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post", "2023-11-05T17:07:13Z"), _s1("pre", "2023-10-14T17:07:13Z"),
            _item("s2_b", "sentinel-2-l2a", "2023-11-06T10:20:00Z", **{"eo:cloud_cover": 12.0}),
            _item("s2_a", "sentinel-2-l2a", "2023-11-04T10:20:00Z", **{"eo:cloud_cover": 85.0}),
            _item("s2_outside", "sentinel-2-l2a", "2023-11-09T10:20:00Z", **{"eo:cloud_cover": 1.0}),
            _item("dem_2", "cop-dem-glo-30", "2021-04-22T00:00:00Z"),
            _item("dem_1", "cop-dem-glo-30", "2021-04-22T00:00:00Z"),
        ])
        m = search_matched_pairs(_query(), PC, client=cat)
        assert m.s2_ids == ("s2_a", "s2_b") and m.s2_cloud_cover == (85.0, 12.0)
        assert m.dem_ids == ("dem_1", "dem_2") and m.catalogue == "planetary_computer"

    def test_reversed_window_is_rejected(self) -> None:
        q = EventQuery("e0", (10.0, 45.0, 10.5, 45.5), datetime(2023, 11, 7), datetime(2023, 11, 4))
        with pytest.raises(ValueError):
            search_matched_pairs(q, PC, client=InMemoryCatalogue())


class TestSentinel2Dedupe:
    def test_reprocessed_duplicates_collapse_to_the_newest_baseline(self) -> None:
        cat = InMemoryCatalogue.from_dicts([
            _s1("post", "2023-11-05T17:07:13Z"), _s1("pre", "2023-10-14T17:07:13Z"),
            _item("S2B_MSIL2A_20231107T102139_R065_T32TPQ_20231107T152044", "sentinel-2-l2a",
                  "2023-11-07T10:21:39Z", **{"eo:cloud_cover": 0.78, "s2:processing_baseline": "05.09"}),
            _item("S2B_MSIL2A_20231107T102139_R065_T32TPQ_20241031T225802", "sentinel-2-l2a",
                  "2023-11-07T10:21:39Z", **{"eo:cloud_cover": 0.79, "s2:processing_baseline": "05.10"}),
            _item("S2B_MSIL2A_20231107T102139_R065_T32TPR_20231107T152052", "sentinel-2-l2a",
                  "2023-11-07T10:21:39Z", **{"eo:cloud_cover": 30.4, "s2:processing_baseline": "05.09"}),
        ])
        m = search_matched_pairs(_query(), PC, client=cat)
        assert m.s2_ids == ("S2B_MSIL2A_20231107T102139_R065_T32TPQ_20241031T225802",
                            "S2B_MSIL2A_20231107T102139_R065_T32TPR_20231107T152052")

    def test_tiles_of_one_pass_count_as_one_overpass(self) -> None:
        m = MatchedScenes("e0", "a", "b", 1, "ascending", ("t1", "t2", "t3"), (0.8, 30.4, 5.0), (),
                          s2_datetimes=("2023-11-07T10:21:39+00:00", "2023-11-07T10:21:39+00:00",
                                        "2023-11-10T10:31:00+00:00"))
        out = summarise_optical_availability((m,))
        assert out["overpasses"] == 2 and out["usable"] == 2  # pass 1: mean 15.6 % < 20 %


class TestPersistence:
    def test_round_trip(self, tmp_path) -> None:
        m = MatchedScenes("e0", "post", "pre", 117, "ascending", ("s2",), (10.0,), ("dem",),
                          ("post",), ("pre",))
        save_matched_scenes([m], tmp_path / "m.json")
        assert load_matched_scenes(tmp_path / "m.json") == (m,)

    def test_catalogue_replay(self, tmp_path) -> None:
        cat = InMemoryCatalogue.from_dicts([_s1("post", "2023-11-05T17:07:13Z"), _s1("pre", "2023-10-14T17:07:13Z")])
        cat.save(tmp_path / "cat.json")
        replay = InMemoryCatalogue.load(tmp_path / "cat.json")
        assert search_matched_pairs(_query(), PC, client=replay).s1_pre_id == "pre"

    def test_saved_catalogue_drops_signing_tokens(self, tmp_path) -> None:
        item = _s1("a", "2023-11-05T17:07:13Z")
        item["assets"] = {"vv": {"href": "https://x.blob.core.windows.net/c/vv.tif?st=1&sig=SECRET"}}
        InMemoryCatalogue.from_dicts([item]).save(tmp_path / "c.json")
        text = (tmp_path / "c.json").read_text()
        assert "SECRET" not in text and "vv.tif" in text

    def test_redact(self) -> None:
        from floodfusion.data.stac import redact

        assert "sig=" not in redact("error for url: https://a.blob.core.windows.net/x.tif?se=2&sig=abc")

    def test_search_by_ids(self) -> None:
        cat = InMemoryCatalogue.from_dicts([_s1("a", "2023-11-05T17:07:13Z"), _s1("b", "2023-11-05T17:07:13Z")])
        assert [i["id"] for i in cat.search(ids=["b"])] == ["b"]


class TestOpticalAvailability:
    def test_empty_is_nan(self) -> None:
        out = summarise_optical_availability(())
        assert math.isnan(out["p_clear"]) and out["n_for_90pct"] == math.inf

    def test_binomial_overpass_count(self) -> None:
        m = MatchedScenes("e0", "a", "b", 1, "ascending", ("x", "y", "z", "w"), (5.0, 50.0, 90.0, float("nan")), ())
        out = summarise_optical_availability((m,))
        assert out["p_clear"] == pytest.approx(0.25)
        assert out["n_for_90pct"] == math.ceil(math.log(0.1) / math.log(0.75))


class TestNamedAcquisition:
    """A label drawn on one named acquisition (Sen1Floods11) pins the orbit and the S2 date."""

    def _catalogue(self) -> InMemoryCatalogue:
        return InMemoryCatalogue.from_dicts([
            _s1("post_168", "2023-11-05T05:20:00Z", 168, "descending"),
            _s1("post_117", "2023-11-07T17:07:13Z"),
            _s1("pre_168", "2023-10-24T05:20:00Z", 168, "descending"),
            _s1("pre_117", "2023-10-31T17:07:13Z"),
            _item("S2_flood_window", "sentinel-2-l2a", "2023-11-06T10:21:39Z", **{"eo:cloud_cover": 5.0}),
            _item("S2_named_date", "sentinel-2-l2a", "2023-11-09T10:21:39Z", **{"eo:cloud_cover": 60.0}),
        ])

    def test_requested_orbit_overrides_the_earliest_pass(self) -> None:
        m = search_matched_pairs(_query(relative_orbit=117, orbit_state="ascending"), PC, client=self._catalogue())
        assert (m.s1_post_id, m.s1_pre_id, m.relative_orbit) == ("post_117", "pre_117", 117)

    def test_no_pass_on_the_requested_orbit_raises(self) -> None:
        with pytest.raises(LookupError, match="relative orbit 15"):
            search_matched_pairs(_query(relative_orbit=15), PC, client=self._catalogue())

    def test_orbit_state_alone_filters_the_pass(self) -> None:
        m = search_matched_pairs(_query(orbit_state="ASCENDING"), PC, client=self._catalogue())
        assert m.s1_post_id == "post_117"

    def test_optical_window_is_independent_and_day_inclusive(self) -> None:
        cat = self._catalogue()
        assert search_matched_pairs(_query(), PC, client=cat).s2_ids == ("S2_flood_window",)
        named = _query(optical_start=datetime(2023, 11, 9), optical_end=datetime(2023, 11, 9))
        assert search_matched_pairs(named, PC, client=cat).s2_ids == ("S2_named_date",)

    def test_reversed_optical_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="optical_end"):
            search_matched_pairs(_query(optical_start=datetime(2023, 11, 9), optical_end=datetime(2023, 11, 8)),
                                 PC, client=self._catalogue())
