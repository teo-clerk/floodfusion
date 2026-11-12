"""floodfusion Live's watcher (``floodfusion.watch``), offline: what it maps, what it never requests, and its index."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from floodfusion import live_rules, watch
from floodfusion.data.stac import Pass

NOW = datetime(2026, 11, 11, 6, 0, tzinfo=UTC)
FORLI = {
    "code": "EMSR664", "name": "Flood in Italy", "category": "Flood", "eventTime": "2023-05-16T11:00:00",
    "aois": [{"number": "1", "name": "Forlì", "extent": "POLYGON ((12.0 44.2, 12.1 44.2, 12.1 44.3, 12.0 44.3, 12.0 44.2))",
              "products": [{"type": "DEL", "downloadPath": "https://example.invalid/never-read.zip"}]}],
}
PASSES = [Pass(datetime(2023, 5, 17, 5, 16, tzinfo=UTC), 95, "descending"),
          Pass(datetime(2023, 5, 18, 17, 5, tzinfo=UTC), 117, "ascending")]


def _config(tmp_path: Path, *, blind: bool = False, showcase: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "live.yaml"
    path.write_text(json.dumps({"blind_floods": blind, "showcase": list(showcase)}), encoding="utf-8")
    return path


class Fake:
    """Records every request; maps by writing what ``floodfusion.live.map_area`` writes."""

    def __init__(self, records: dict | None = None, passes: list[Pass] | None = None, fail: set[str] = frozenset()):
        self.records, self.passes, self.fail = records or {}, passes if passes is not None else PASSES, fail
        self.fetched: list[str] = []
        self.searched: list[tuple] = []
        self.mapped: list[str] = []

    def fetch(self, code: str) -> dict:
        self.fetched.append(code)
        if code not in self.records:
            raise LookupError(code)
        return self.records[code]

    def passes_of(self, bbox, start, end) -> list[Pass]:
        self.searched.append((tuple(bbox), start, end))
        return [p for p in self.passes if start <= p.time <= end]

    def mapper(self, job: watch.MapJob, out: Path) -> dict:
        self.mapped.append(job.map_id)
        if job.map_id in self.fail:
            raise RuntimeError("Planetary Computer timed out")
        tile = out / "maps" / "t00_00"
        tile.mkdir(parents=True)
        (tile / "flood.geojson").write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
        (tile / "item.json").write_text("{}", encoding="utf-8")
        (tile / "likelihood.tif").write_bytes(b"COG")
        summary = {"tiles": {"t00_00": {"flood_km2": 1.5, "valid_km2": 40.0, "pass_time": job.pass_.time.isoformat()}},
                   "skipped": {}}
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary


def _run(tmp_path: Path, fake: Fake, config: Path, **kw) -> dict:
    return watch.run(tmp_path / "data", tmp_path / "work", config=config, fetch=fake.fetch, passes_of=fake.passes_of,
                     mapper=fake.mapper, now=kw.pop("now", NOW), **kw)


def test_with_blind_floods_off_and_no_showcase_nothing_is_requested(tmp_path: Path) -> None:
    fake = Fake()
    index = _run(tmp_path, fake, _config(tmp_path))
    assert fake.fetched == [] and fake.searched == [] and fake.mapped == []
    assert index["maps"] == [] and index["blind_floods"] is False
    assert json.loads((tmp_path / "data" / "index.json").read_text())["maps"] == []


def test_a_seen_showcase_flood_is_mapped_once_per_pass_and_never_twice(tmp_path: Path) -> None:
    fake = Fake(records={"EMSR664": FORLI})
    config = _config(tmp_path, showcase=("EMSR664",))
    index = _run(tmp_path, fake, config)
    assert [m["id"] for m in index["maps"]] == ["EMSR664_AOI01_20230517T0516_095", "EMSR664_AOI01_20230518T1705_117"]
    first = index["maps"][0]
    assert first["kind"] == "showcase" and first["aoi_name"] == "Forlì" and first["flood_km2"] == 1.5
    assert first["bbox"] == [12.0, 44.2, 12.1, 44.3] and first["pass_time"] == "2023-05-17T05:16:00+00:00"
    published = tmp_path / "data" / "maps" / first["id"]
    assert sorted(p.relative_to(published).as_posix() for p in published.rglob("*") if p.is_file()) == [
        "summary.json", "t00_00/flood.geojson", "t00_00/item.json"]      # the COGs are not published here
    assert first["files"] == [f"maps/{first['id']}/t00_00/flood.geojson"]
    assert fake.searched[0][1:] == (datetime(2023, 5, 16, tzinfo=UTC), datetime(2023, 5, 19, 23, 59, 59, tzinfo=UTC))

    again = Fake(records={"EMSR664": FORLI})
    index = _run(tmp_path, again, config)
    assert len(index["maps"]) == 2 and again.mapped == [] and again.fetched == []   # complete: not even re-read


def test_a_blind_showcase_code_is_refused_before_its_record_is_requested(tmp_path: Path) -> None:
    fake = Fake(records={"EMSR926": {**FORLI, "code": "EMSR926"}})
    with pytest.raises(live_rules.BlindFloodRefusedError):
        _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR926",)))
    assert fake.fetched == []


def test_passes_that_reach_the_blind_dates_are_not_mapped(tmp_path: Path) -> None:
    late = {**FORLI, "code": "EMSR898", "eventTime": "2026-07-16T00:00:00"}
    passes = [Pass(datetime(2026, 7, 17, 5, 0, tzinfo=UTC), 95, "descending"),
              Pass(datetime(2026, 7, 18, 5, 0, tzinfo=UTC), 95, "descending")]
    fake = Fake(records={"EMSR898": late}, passes=passes)
    index = _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR898",)))
    assert fake.mapped == ["EMSR898_AOI01_20260717T0500_095"] and len(index["maps"]) == 1


def test_a_failed_map_does_not_stop_the_others_and_is_tried_again(tmp_path: Path) -> None:
    config = _config(tmp_path, showcase=("EMSR664",))
    fake = Fake(records={"EMSR664": FORLI}, fail={"EMSR664_AOI01_20230517T0516_095"})
    index = _run(tmp_path, fake, config)
    assert [m["id"] for m in index["maps"]] == ["EMSR664_AOI01_20230518T1705_117"]
    assert "EMSR664" not in index["complete"]
    again = Fake(records={"EMSR664": FORLI})
    index = _run(tmp_path, again, config)
    assert again.mapped == ["EMSR664_AOI01_20230517T0516_095"] and "EMSR664" in index["complete"]


def test_new_activations_are_watched_only_once_blind_floods_are_on(tmp_path: Path) -> None:
    flood = {**FORLI, "code": "EMSR950", "eventTime": "2026-11-09T00:00:00"}
    fire = {**FORLI, "code": "EMSR951", "category": "Wildfire", "eventTime": "2026-11-09T00:00:00"}
    passes = [Pass(datetime(2026, 11, 10, 5, 0, tzinfo=UTC), 95, "descending")]
    fake = Fake(records={"EMSR950": flood, "EMSR951": fire}, passes=passes)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "index.json").write_text(json.dumps({"maps": [], "complete": [], "last_code": 949}))
    index = _run(tmp_path, fake, _config(tmp_path, blind=True), stop_after=2)
    assert fake.fetched == ["EMSR950", "EMSR951", "EMSR952", "EMSR953"] and index["last_code"] == 951
    assert [m["id"] for m in index["maps"]] == ["EMSR950_AOI01_20261110T0500_095"]
    live_map = index["maps"][0]
    assert live_map["kind"] == "live" and live_map["delay_hours"] == 25.0
    assert fake.searched[0][2] == NOW                   # a live flood: every pass up to now


def test_a_manual_map_takes_the_passes_of_its_date_and_refuses_a_blind_date(tmp_path: Path) -> None:
    fake = Fake()
    index = _run(tmp_path, fake, _config(tmp_path), manual=((12.0, 44.2, 12.1, 44.3), date(2023, 5, 17)))
    assert [m["kind"] for m in index["maps"]] == ["manual"] and index["maps"][0]["code"] is None
    assert index["maps"][0]["id"].startswith("manual_20230517T0516_095_")
    with pytest.raises(live_rules.BlindFloodRefusedError):
        _run(tmp_path, Fake(), _config(tmp_path), manual=((12.0, 44.2, 12.1, 44.3), date(2026, 10, 1)))


def test_the_runs_per_call_are_capped(tmp_path: Path) -> None:
    many = [Pass(datetime(2023, 5, 16, h, 0, tzinfo=UTC), 95, "descending") for h in range(10)]
    fake = Fake(records={"EMSR664": FORLI}, passes=many)
    index = _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR664",)))
    assert len(index["maps"]) == watch.MAX_MAPS_PER_RUN and "EMSR664" not in index["complete"]


def test_the_command_needs_a_bbox_and_a_date_together() -> None:
    with pytest.raises(SystemExit):
        watch.main(["run", "--data", "d", "--work", "w", "--bbox", "12.0", "44.2", "12.1", "44.3"])


def test_a_run_can_be_capped_below_the_default(tmp_path: Path) -> None:
    fake = Fake(records={"EMSR664": FORLI})
    index = _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR664",)), max_maps=1)
    assert len(fake.mapped) == 1 and len(index["maps"]) == 1


def test_the_committed_showcase_is_seen_activations_only() -> None:
    import yaml

    # 2026-11-12: the file moved to live_rules.LIVE_CONFIG (configs/live/service.yaml); the check is unchanged.
    showcase = yaml.safe_load((Path(__file__).resolve().parents[1] / live_rules.LIVE_CONFIG).read_text())["showcase"]
    assert showcase and all(live_rules.check_code(code, enabled=False) for code in showcase)


# --- after the code review of 2026-11-11 ---------------------------------------------------------------------------

class TileFake(Fake):
    """A mapper whose tiles all fail or all have no pass, as map_area reports them."""

    def __init__(self, *args, skipped: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped = skipped

    def mapper(self, job, out):
        self.mapped.append(job.map_id)
        out.mkdir(parents=True)
        summary = {"tiles": {}, "skipped": {"t00_00": f"{self.skipped}: Planetary Computer timed out"}}
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary


def test_a_map_whose_tiles_all_failed_is_not_published_and_is_tried_again(tmp_path: Path) -> None:
    config = _config(tmp_path, showcase=("EMSR664",))
    index = _run(tmp_path, TileFake(records={"EMSR664": FORLI}, skipped="failed"), config)
    assert index["maps"] == [] and index["failures"] == {"EMSR664_AOI01_20230517T0516_095": 1,
                                                         "EMSR664_AOI01_20230518T1705_117": 1}
    again = Fake(records={"EMSR664": FORLI})
    assert len(_run(tmp_path, again, config)["maps"]) == 2 and again.mapped


def test_a_map_with_no_usable_pass_on_any_tile_is_recorded_and_not_tried_again(tmp_path: Path) -> None:
    config = _config(tmp_path, showcase=("EMSR664",))
    index = _run(tmp_path, TileFake(records={"EMSR664": FORLI}, skipped="dropped"), config)
    assert index["maps"] == [] and len(index["empty"]) == 2 and "EMSR664" in index["complete"]


def test_a_killed_run_keeps_the_maps_it_made_and_counts_the_attempt(tmp_path: Path) -> None:
    class Killed(Fake):
        def mapper(self, job, out):
            if self.mapped:
                raise KeyboardInterrupt            # the job's timeout
            return super().mapper(job, out)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, Killed(records={"EMSR664": FORLI}), _config(tmp_path, showcase=("EMSR664",)))
    saved = json.loads((tmp_path / "data" / "index.json").read_text())
    assert [m["id"] for m in saved["maps"]] == ["EMSR664_AOI01_20230517T0516_095"]
    assert saved["failures"] == {"EMSR664_AOI01_20230518T1705_117": 1}


def test_a_map_that_keeps_failing_is_abandoned_and_listed(tmp_path: Path) -> None:
    config = _config(tmp_path, showcase=("EMSR664",))
    bad = "EMSR664_AOI01_20230517T0516_095"
    for _ in range(watch.MAX_ATTEMPTS):
        index = _run(tmp_path, Fake(records={"EMSR664": FORLI}, fail={bad}), config)
    assert index["abandoned"] == [bad] and "EMSR664" in index["complete"] and bad not in index["failures"]


def test_a_record_that_cannot_be_fetched_does_not_stop_the_run(tmp_path: Path) -> None:
    class Flaky(Fake):
        def fetch(self, code):
            if code == "EMSR756":
                raise ConnectionError("rapidmapping.emergency.copernicus.eu timed out")
            return super().fetch(code)

    fake = Flaky(records={"EMSR664": FORLI})
    index = _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR756", "EMSR664")))
    assert len(index["maps"]) == 2 and "EMSR756" not in index["complete"]


def test_a_code_not_public_yet_is_asked_for_again_once_a_later_one_is(tmp_path: Path) -> None:
    flood = {**FORLI, "eventTime": "2026-11-09T00:00:00"}
    passes = [Pass(datetime(2026, 11, 10, 5, 0, tzinfo=UTC), 95, "descending")]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "index.json").write_text(json.dumps({"last_code": 949}))
    config = _config(tmp_path, blind=True)
    first = Fake(records={"EMSR951": {**flood, "code": "EMSR951"}}, passes=passes)
    index = _run(tmp_path, first, config, stop_after=2)
    assert index["last_code"] == 951 and list(index["pending_codes"]) == ["EMSR950"]
    later = Fake(records={"EMSR950": {**flood, "code": "EMSR950"}, "EMSR951": {**flood, "code": "EMSR951"}},
                 passes=passes)
    index = _run(tmp_path, later, config, stop_after=2)
    assert "EMSR950" in index["watching"] and index["pending_codes"] == {}
    assert later.fetched[:2] == ["EMSR951", "EMSR950"]          # the watched code, then the pending one


def test_aois_without_a_number_get_distinct_ids(tmp_path: Path) -> None:
    aoi = FORLI["aois"][0]
    record = {**FORLI, "aois": [{**aoi, "number": None}, {**aoi, "number": None, "name": "Lugo"}]}
    fake = Fake(records={"EMSR664": record}, passes=PASSES[:1])
    index = _run(tmp_path, fake, _config(tmp_path, showcase=("EMSR664",)))
    assert [m["id"] for m in index["maps"]] == ["EMSR664_AOI01_20230517T0516_095", "EMSR664_AOI02_20230517T0516_095"]


def test_a_published_map_records_the_tiles_its_pass_imaged(tmp_path: Path) -> None:
    class WithBoxes(Fake):
        def mapper(self, job, out):
            summary = super().mapper(job, out)
            summary["tiles"]["t00_00"]["bbox"] = [12.0, 44.2, 12.05, 44.25]
            return summary

    index = _run(tmp_path, WithBoxes(records={"EMSR664": FORLI}, passes=PASSES[:1]),
                 _config(tmp_path, showcase=("EMSR664",)))
    assert index["maps"][0]["tile_bboxes"] == [[12.0, 44.2, 12.05, 44.25]]
