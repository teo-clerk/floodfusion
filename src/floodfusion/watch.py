r"""floodfusion Live's watcher: which flood maps to make next, and the index the live page reads.

Each run (GitHub Actions, every 6 hours, ``.github/workflows/live.yml``):

1. **Showcase:** the seen activations listed in ``configs/live.yaml`` (EMSR656–898), each AOI mapped on every
   Sentinel-1 pass from its event date to 3 days after. A code whose window has passed and whose maps are all made is
   *complete* and its record is not requested again.
2. **New activations,** only once blind floods are switched on: codes after the last one seen are requested until
   20 in a row are not public. Every flood activation is then watched for 30 days, each new pass mapped.
3. **A manual map** (``--bbox``/``--date``): every pass over the box on that date.

What may be requested and mapped is :mod:`floodfusion.live_rules`'s: while blind floods are off, no code from EMSR899
on is requested and no pass whose optical window reaches 20 July 2026 is mapped. Of a record only the code, name,
category, event date and AOI extents are read.

A map is made by :func:`floodfusion.live.map_area`. Its light files (``summary.json``, and per tile ``flood.geojson``
and the STAC ``item.json``) are copied into the data folder (the ``live-data`` branch) beside ``index.json``; the
COGs stay in the work folder, kept by the workflow as the run's artifact. At most :data:`MAX_MAPS_PER_RUN` maps
are made per run. A map that fails, or whose tiles all fail, is tried again on the next runs, up to
:data:`MAX_ATTEMPTS` times, then listed as ``abandoned``; a map with no usable pass on any tile is listed as
``empty``. The index is saved after every map, so a job stopped by its timeout keeps what it made.

Run::

    uv run --frozen python -m floodfusion.watch run --data live-data --work work
    uv run --frozen python -m floodfusion.watch run --data live-data --work work --bbox 12.0 44.2 12.1 44.3 \
        --date 2023-05-17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from floodfusion.data.stac import Pass
from floodfusion.live_rules import (
    BLIND_FIRST_CODE,
    LIVE_CONFIG,
    ActivationSummary,
    BlindFloodRefusedError,
    allowed_fields,
    blind_floods_enabled,
    check_code,
    check_date,
)

__all__ = ["MAX_ATTEMPTS", "MAX_MAPS_PER_RUN", "MapFailedError", "MapJob", "main", "run"]

LOGGER = logging.getLogger(__name__)

INDEX = "index.json"
MAX_AOIS = 4
#: Measured on 2026-11-11: about a minute per mapped tile on 4 CPU threads, and an AOI can hold 20+ tiles.
MAX_MAPS_PER_RUN = 2
MAX_ATTEMPTS = 3
SHOWCASE_DAYS = 3
LIVE_DAYS = 30
STOP_AFTER_NON_PUBLIC = 20
_PUBLISHED = ("flood.geojson", "item.json")

Fetch = Callable[[str], Mapping[str, Any]]
PassesOf = Callable[[Sequence[float], datetime, datetime], list[Pass]]


@dataclass(frozen=True)
class MapJob:
    map_id: str
    kind: str                     # showcase | live | manual
    code: str | None
    name: str
    aoi: int | None
    aoi_name: str
    bbox: tuple[float, float, float, float]
    pass_: Pass
    event_date: date | None


def _day(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=UTC)


def _day_end(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=UTC)


def _stamp(p: Pass) -> str:
    return f"{p.time.astimezone(UTC):%Y%m%dT%H%M}_{p.relative_orbit:03d}"


def _allowed(passes: Iterable[Pass], *, enabled: bool) -> list[Pass]:
    out = []
    for p in passes:
        try:
            check_date(p.time.date(), enabled=enabled)
        except BlindFloodRefusedError as exc:
            LOGGER.info("not mapped: %s", exc)
            continue
        out.append(p)
    return out


def _activation_jobs(summary: ActivationSummary, kind: str, until: datetime, passes_of: PassesOf, *,
                     enabled: bool) -> list[MapJob]:
    jobs = []
    for aoi in summary.aois[:MAX_AOIS]:
        for p in _allowed(passes_of(aoi.bbox, _day(summary.event_date), until), enabled=enabled):
            jobs.append(MapJob(map_id=f"{summary.code}_AOI{aoi.number:02d}_{_stamp(p)}", kind=kind, code=summary.code,
                               name=summary.name, aoi=aoi.number, aoi_name=aoi.name, bbox=aoi.bbox, pass_=p,
                               event_date=summary.event_date))
    return jobs


def _try_fetch(code: str, fetch: Fetch) -> Mapping[str, Any] | None:
    """The record, or None (logged) if it cannot be fetched now; ``LookupError`` (not public) propagates."""
    try:
        return fetch(code)
    except LookupError:
        raise
    except Exception as exc:  # noqa: BLE001 - one unreachable record must not stop the run
        LOGGER.warning("%s not fetched this run: %s: %s", code, type(exc).__name__, str(exc)[:200])
        return None


def _showcase(codes: Sequence[str], index: Mapping[str, Any], fetch: Fetch, passes_of: PassesOf, *, enabled: bool,
              now: datetime) -> dict[str, tuple[list[MapJob], bool]]:
    """``{code: (jobs, window_over)}`` for the showcase codes not yet complete."""
    out = {}
    for code in codes:
        check_code(code, enabled=enabled)                  # before any request
        if code in index["complete"]:
            continue
        try:
            record = _try_fetch(code, fetch)
        except LookupError as exc:
            LOGGER.warning("showcase code %s is not public: %s", code, exc)
            continue
        if record is None:
            continue
        summary = allowed_fields(record, enabled=enabled)
        end = _day_end(summary.event_date + timedelta(days=SHOWCASE_DAYS))
        out[code] = (_activation_jobs(summary, "showcase", min(end, now), passes_of, enabled=enabled), now > end)
    return out


def _flood(record: Mapping[str, Any], *, enabled: bool) -> ActivationSummary | None:
    try:
        return allowed_fields(record, enabled=enabled)
    except BlindFloodRefusedError:
        raise
    except ValueError as exc:                               # not a flood activation
        LOGGER.info("%s not watched: %s", str(record.get("code", ""))[:12], exc)
        return None


def _pending(index: dict[str, Any], fetch: Fetch, *, enabled: bool, now: datetime) -> dict[str, ActivationSummary]:
    """Codes skipped while a later one was public, asked for again until public or ``LIVE_DAYS`` old."""
    found = {}
    for code, first_missed in list(index["pending_codes"].items()):
        try:
            record = _try_fetch(check_code(code, enabled=enabled), fetch)
        except LookupError:
            if now - datetime.fromisoformat(first_missed) > timedelta(days=LIVE_DAYS):
                index["pending_codes"].pop(code)
            continue
        if record is not None:
            index["pending_codes"].pop(code)
            summary = _flood(record, enabled=enabled)
            found.update({code: summary} if summary else {})
    return found


def _discover(index: dict[str, Any], fetch: Fetch, *, enabled: bool, now: datetime,
              stop_after: int) -> dict[str, ActivationSummary]:
    """Codes after ``last_code`` until ``stop_after`` in a row are not public; gaps before a public one are pending."""
    found, missed = {}, []
    n = int(index["last_code"]) + 1
    while len(missed) < stop_after:
        code = check_code(f"EMSR{n}", enabled=enabled)
        try:
            record = _try_fetch(code, fetch)
        except LookupError:
            missed.append(code)
            n += 1
            continue
        if record is None:                                  # unreachable: scan again from here next run
            break
        index["last_code"] = n
        for gap in missed:
            index["pending_codes"].setdefault(gap, now.isoformat())
        missed = []
        summary = _flood(record, enabled=enabled)
        found.update({code: summary} if summary else {})
        n += 1
    return found


def _live(index: dict[str, Any], fetch: Fetch, passes_of: PassesOf, *, enabled: bool, now: datetime,
          stop_after: int) -> dict[str, tuple[list[MapJob], bool]]:
    """New and watched activations; nothing at all is requested while blind floods are off."""
    if not enabled:
        return {}
    summaries = {}
    for code in list(index["watching"]):
        try:
            record = _try_fetch(check_code(code, enabled=enabled), fetch)
        except LookupError as exc:
            LOGGER.warning("watched code %s is not public any more: %s", code, exc)
            continue
        summary = _flood(record, enabled=enabled) if record is not None else None
        summaries.update({code: summary} if summary else {})
    summaries.update(_pending(index, fetch, enabled=enabled, now=now))
    summaries.update(_discover(index, fetch, enabled=enabled, now=now, stop_after=stop_after))
    out = {}
    for code, summary in summaries.items():
        end = _day_end(summary.event_date + timedelta(days=LIVE_DAYS))
        index["watching"][code] = summary.event_date.isoformat()
        out[code] = (_activation_jobs(summary, "live", min(end, now), passes_of, enabled=enabled), now > end)
    return out


def _manual(bbox: Sequence[float], when: date, passes_of: PassesOf, *, enabled: bool) -> list[MapJob]:
    from floodfusion.live import check_bbox

    check_date(when, enabled=enabled)
    box = check_bbox(bbox)
    digest = hashlib.sha256(json.dumps(box).encode()).hexdigest()[:8]
    return [MapJob(map_id=f"manual_{_stamp(p)}_{digest}", kind="manual", code=None, name="manual", aoi=None,
                   aoi_name="", bbox=box, pass_=p, event_date=None)
            for p in _allowed(passes_of(box, _day(when), _day_end(when)), enabled=enabled)]


class MapFailedError(RuntimeError):
    """Every tile of a map failed (for example, the catalogue was unreachable): the map is tried again."""


def _load_index(root: Path) -> dict[str, Any]:
    path = root / INDEX
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"maps": [], "complete": [], "watching": {}, "failures": {}, "abandoned": [], "empty": [],
            "pending_codes": {}, "last_code": BLIND_FIRST_CODE - 1, **index}


def _publish(job: MapJob, summary: Mapping[str, Any], work: Path, data_root: Path, now: datetime) -> dict[str, Any]:
    target = data_root / "maps" / job.map_id
    files = []
    for tile_id in sorted(summary.get("tiles", {})):
        for name in _PUBLISHED:
            source = work / "maps" / tile_id / name
            if source.is_file():
                (target / tile_id).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target / tile_id / name)
                if name == "flood.geojson":
                    files.append(f"maps/{job.map_id}/{tile_id}/{name}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "summary.json", target / "summary.json")
    tiles = summary.get("tiles", {}).values()
    return {"id": job.map_id, "kind": job.kind, "code": job.code, "name": job.name, "aoi": job.aoi,
            "aoi_name": job.aoi_name, "bbox": list(job.bbox), "event_date": job.event_date.isoformat() if job.event_date
            else None, "pass_time": job.pass_.time.isoformat(), "orbit": job.pass_.relative_orbit,
            "orbit_state": job.pass_.orbit_state, "mapped_at": now.isoformat(),
            "delay_hours": round((now - job.pass_.time).total_seconds() / 3600, 1),
            "flood_km2": round(sum(t["flood_km2"] for t in tiles), 4),
            "valid_km2": round(sum(t["valid_km2"] for t in tiles), 4), "tiles": len(summary.get("tiles", {})),
            "skipped_tiles": len(summary.get("skipped", {})), "files": files,
            "tile_bboxes": [list(t["bbox"]) for _, t in sorted(summary.get("tiles", {}).items()) if "bbox" in t]}


def _checked(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if not summary.get("tiles") and any(str(v).startswith("failed:") for v in summary.get("skipped", {}).values()):
        raise MapFailedError(f"no tile mapped: {next(iter(summary['skipped'].values()))[:200]}")
    return summary


def _attempt(job: MapJob, index: dict[str, Any], mapper: Callable[[MapJob, Path], Mapping[str, Any]], work: Path,
             data_root: Path, now: datetime, save: Callable[[], None]) -> None:
    """One map. The attempt is counted and saved before it starts, so a killed job still counts it."""
    attempts = index["failures"].get(job.map_id, 0) + 1
    index["failures"][job.map_id] = attempts
    save()
    try:
        summary = _checked(mapper(job, work))
    except Exception as exc:  # noqa: BLE001 - one map must not stop the others; it is tried again
        LOGGER.warning("%s failed (attempt %d of %d): %s: %s", job.map_id, attempts, MAX_ATTEMPTS,
                       type(exc).__name__, str(exc)[:300])
        if attempts >= MAX_ATTEMPTS:
            index["failures"].pop(job.map_id)
            index["abandoned"].append(job.map_id)
        save()
        return
    index["failures"].pop(job.map_id)
    if summary["tiles"]:
        index["maps"].append(_publish(job, summary, work, data_root, now))
    else:                                                   # no usable pass on any tile: nothing to show
        index["empty"].append(job.map_id)
    save()


def run(data_root: Path, work_root: Path, *, config: Path = LIVE_CONFIG, fetch: Fetch, passes_of: PassesOf,
        mapper: Callable[[MapJob, Path], Mapping[str, Any]], now: datetime | None = None,
        manual: tuple[Sequence[float], date] | None = None, stop_after: int = STOP_AFTER_NON_PUBLIC,
        max_maps: int = MAX_MAPS_PER_RUN) -> dict[str, Any]:
    """One watcher run; returns the updated index, saved to ``data_root/index.json`` after every map."""
    import yaml

    from floodfusion.data.records import atomic_write_bytes

    now = now or datetime.now(UTC)
    enabled = blind_floods_enabled(config)
    showcase = [str(c) for c in (yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}).get("showcase", [])]
    data_root, work_root = Path(data_root), Path(work_root)
    index = _load_index(data_root)

    def save() -> None:
        index.update(blind_floods=enabled, updated_at=now.isoformat())
        data_root.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(data_root / INDEX, json.dumps(index, indent=1, sort_keys=True).encode("utf-8"))

    def unfinished(jobs: Sequence[MapJob]) -> list[MapJob]:
        settled = {m["id"] for m in index["maps"]} | set(index["empty"]) | set(index["abandoned"])
        return [j for j in jobs if j.map_id not in settled]

    groups = {}
    if manual is not None:
        groups["manual"] = (_manual(*manual, passes_of, enabled=enabled), False)
    groups.update(_showcase(showcase, index, fetch, passes_of, enabled=enabled, now=now))
    groups.update(_live(index, fetch, passes_of, enabled=enabled, now=now, stop_after=stop_after))
    budget = max_maps
    for code, (jobs, window_over) in groups.items():
        for job in unfinished(jobs)[:budget]:
            budget -= 1
            _attempt(job, index, mapper, work_root / job.map_id, data_root, now, save)
        if code != "manual" and window_over and not unfinished(jobs) and code not in index["complete"]:
            index["complete"].append(code)
            index["watching"].pop(code, None)
    save()
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="floodfusion Live's watcher: the next flood maps.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("run")
    one.add_argument("--data", type=Path, required=True, help="the published folder (the live-data branch)")
    one.add_argument("--work", type=Path, required=True, help="where maps are made (COGs stay here)")
    one.add_argument("--config", type=Path, default=LIVE_CONFIG)
    one.add_argument("--run", type=Path, default=None, help="a frozen run folder (default: floodfusion.live's)")
    one.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    one.add_argument("--date", type=date.fromisoformat, default=None, help="with --bbox: a manual map's date")
    one.add_argument("--max-maps", type=int, default=MAX_MAPS_PER_RUN, help="maps made in this run, at most")
    args = parser.parse_args(argv)
    if (args.bbox is None) != (args.date is None):
        parser.error("--bbox and --date go together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from floodfusion import live
    from floodfusion.config import load_config
    from floodfusion.data.archive import IngestSources
    from floodfusion.data.cems import fetch_activation
    from floodfusion.data.stac import PROFILES, list_passes
    from floodfusion.train import resolve_device

    sources = IngestSources.open()
    profile = PROFILES["planetary_computer"]
    cfg = load_config(Path("configs/base.yaml"))
    device = resolve_device("cpu")
    predictor = live.load_predictor(args.run or live.DEFAULT_RUN, device=device)

    def mapper(job: MapJob, out: Path) -> Mapping[str, Any]:
        return live.map_area(job.bbox, job.pass_.time.date(), out, predictor=predictor, name=job.map_id, cfg=cfg,
                             sources=sources, device=device, relative_orbit=job.pass_.relative_orbit,
                             orbit_state=job.pass_.orbit_state, reference_before=job.event_date)

    index = run(args.data, args.work, config=args.config, fetch=fetch_activation,
                passes_of=lambda bbox, start, end: list_passes(sources.client, bbox, start, end, profile=profile),
                mapper=mapper, manual=(args.bbox, args.date) if args.bbox else None, max_maps=args.max_maps)
    LOGGER.info("%d maps published; blind floods %s", len(index["maps"]), "on" if index["blind_floods"] else "off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
