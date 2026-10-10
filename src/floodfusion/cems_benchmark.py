r"""Copernicus EMS flood maps as an independent multi-event benchmark, selected by a fixed rule.

Sen1Floods11's labels were drawn on the Sentinel-2 image the fusion model also sees, and Kuro
Siwo's on the Sentinel-1 image every model sees. A CEMS delineation drawn on **another
satellite** (COSMO-SkyMed, PAZ, RADARSAT-2, TerraSAR-X, ICEYE; or Pléiades, WorldView, SPOT)
is independent of both. This module turns the public Rapid Mapping archive into a benchmark:

``eligibility``
    Every flood activation in a code range is checked against a rule fixed before any model
    is scored on it (``reports/protocol_v1_1.md``):

    1. a **delineation** (``DEL``) product whose source images are **all** non-Sentinel;
    2. a Sentinel-1 RTC acquisition within :data:`WINDOW_H` hours of the product's latest
       image, covering at least :data:`MIN_COVERAGE` of the product's extent;
    3. a same-orbit pre-event acquisition in the :data:`PRE_EVENT_WINDOW_DAYS` days before
       the **reference date**: :data:`REFERENCE_GAP_DAYS` before the activation's event
       time, or before an earlier overlapping flood activation's event time if one started
       within :data:`RELATED_LOOKBACK_DAYS` (Emilia-Romagna flooded on 2-3 May 2023 and again
       on 16-17 May: EMSR659 then EMSR664).

    Per AOI the pair (product, pass) with the smallest time offset is kept (ties: the earlier
    pass). The **activation** is the statistical unit; its AOIs pool.

The stratum of a product is the kind of sensor it was drawn on: ``sar`` products favour radar
models (the conservative test of fusion, hence primary), ``optical`` ones favour optical.

Products: © European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from floodfusion.data.passes import Acquisition, acquisitions
from floodfusion.data.passes import coverage as _coverage

__all__ = [
    "MIN_COVERAGE",
    "PRE_EVENT_WINDOW_DAYS",
    "REFERENCE_GAP_DAYS",
    "WINDOW_H",
    "Acquisition",
    "Candidate",
    "acquisitions",
    "cached_activations",
    "eligible_pairs",
    "main",
    "pair_key",
    "prepare_pair",
    "product_stratum",
    "reference_date",
    "sample_tiles",
    "select_pairs",
    "sensor_kind",
]

LOGGER = logging.getLogger(__name__)

#: A Sentinel-1 pass counts when it is this close in time to the product's latest image.
WINDOW_H = 24.0
#: ... and covers at least this fraction of the product's extent.
MIN_COVERAGE = 0.5
#: The pre-event reference must be acquired at least this long before the event time.
REFERENCE_GAP_DAYS = 7
#: How far back the pre-event reference may lie (the ingest's own window).
PRE_EVENT_WINDOW_DAYS = 60
#: An earlier overlapping flood activation this recent moves the reference date back.
RELATED_LOOKBACK_DAYS = 60

#: Sensor-name prefixes as the Rapid Mapping API spells them.
_SAR_PREFIXES = ("COSMO-SkyMed", "PAZ", "RADARSAT", "TerraSAR-X", "TanDEM-X", "ICEYE", "SAOCOM", "Capella",
                 "ALOS", "RCM", "KOMPSAT-5", "NovaSAR", "Umbra")
_OPTICAL_PREFIXES = ("Pleiades", "WorldView", "GeoEye", "SPOT", "PlanetScope", "Landsat", "GEOSAT", "KOMPSAT-3",
                     "SuperView", "Plane", "Aerial", "Deimos", "Kompsat-3", "TripleSat", "Jilin", "Gaofen", "Vision-1")


def sensor_kind(name: str) -> str:
    """``sentinel``, ``sar``, ``optical`` or ``unknown`` for one sensor name from the API."""
    text = str(name or "").strip()
    if text.lower().startswith("sentinel"):
        return "sentinel"
    if text.startswith(_SAR_PREFIXES):
        return "sar"
    if text.startswith(_OPTICAL_PREFIXES):
        return "optical"
    return "unknown"


def product_stratum(sensors: Sequence[str]) -> str | None:
    """``sar`` / ``optical`` / ``mixed`` for a product drawn on non-Sentinel images; ``None`` otherwise.

    A product is ineligible (``None``) if it lists no image, any Sentinel image (it would not
    be independent of the inputs), or an image of unknown kind (it could not be stratified).
    """
    kinds = {sensor_kind(s) for s in sensors}
    if not kinds or kinds & {"sentinel", "unknown"}:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _time(text: str) -> datetime:
    when = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def _geometry(wkt_or_geojson: Any) -> Any:
    import shapely
    from shapely.geometry import shape

    if isinstance(wkt_or_geojson, Mapping):
        return shape(wkt_or_geojson)
    return shapely.from_wkt(str(wkt_or_geojson))


def reference_date(activation: Mapping[str, Any], others: Sequence[Mapping[str, Any]]) -> date:
    """The date the pre-event reference must precede (rule 3 of the module docstring)."""
    event = _time(activation["eventTime"])
    extent = _geometry(activation["extent"])
    cutoff = event - timedelta(days=REFERENCE_GAP_DAYS)
    for other in others:
        if other.get("code") == activation.get("code") or str(other.get("category", "")).lower() != "flood":
            continue
        when = _time(other["eventTime"])
        if event - timedelta(days=RELATED_LOOKBACK_DAYS) <= when < event and _geometry(other["extent"]).intersects(extent):
            cutoff = min(cutoff, when - timedelta(days=REFERENCE_GAP_DAYS))
    return cutoff.date()


@dataclass(frozen=True)
class Candidate:
    """One eligible (product, pass) pair of one AOI."""

    activation: str
    aoi: int
    product: str                 # tag, e.g. EMSR664_AOI01_DEL_MONIT03
    stratum: str                 # sar | optical | mixed
    sensors: tuple[str, ...]
    product_time: str            # latest source image, ISO-8601 UTC
    pass_time: str
    relative_orbit: int
    orbit_state: str
    offset_h: float              # pass - product
    coverage: float
    reference_before: str        # the pre-event reference precedes this date
    n_reference_scenes: int


def eligible_pairs(activation: Mapping[str, Any], client: Any, *,
                   others: Sequence[Mapping[str, Any]] = ()) -> list[Candidate]:
    """The eligible (product, pass) pair of every AOI of one activation, or none."""
    from floodfusion.data.cems import list_products

    if str(activation.get("category", "")).lower() != "flood":
        return []
    ref_day = reference_date(activation, others)
    ref_end = datetime.combine(ref_day, datetime.min.time(), tzinfo=UTC)
    aoi_extents = {int(a.get("number") or 0): a.get("extent") for a in activation.get("aois", [])}
    references: dict[str, list[Acquisition]] = {}   # pre-event acquisitions, searched once per extent
    out: list[Candidate] = []
    by_aoi: dict[int, list[Any]] = {}
    for product in list_products(activation, product_type="DEL"):
        by_aoi.setdefault(product.aoi, []).append(product)
    for aoi, products in sorted(by_aoi.items()):
        best: Candidate | None = None
        for product in products:
            stratum = product_stratum([s for s, _ in product.images])
            if stratum is None or not product.acquisition_times or not product.download_url:
                continue   # not independent, undated, or no published package to score against
            extent_wkt = _product_extent(activation, product) or aoi_extents.get(aoi)
            if not extent_wkt:
                continue
            extent = _geometry(extent_wkt)
            t = max(product.acquisition_times)
            window = timedelta(hours=WINDOW_H)
            for acq in acquisitions(client, extent, t - window, t + window):
                cover = _coverage(extent, acq.footprint)
                if cover < MIN_COVERAGE:
                    continue
                key = str(extent_wkt)
                if key not in references:
                    references[key] = acquisitions(client, extent, ref_end - timedelta(days=PRE_EVENT_WINDOW_DAYS),
                                                   ref_end - timedelta(seconds=1))
                refs = [a for a in references[key]
                        if a.relative_orbit == acq.relative_orbit and a.orbit_state == acq.orbit_state]
                if not refs:
                    continue
                cand = Candidate(product.activation, aoi, product.tag, stratum, tuple(s for s, _ in product.images),
                                 t.isoformat(), acq.when.isoformat(), acq.relative_orbit, acq.orbit_state,
                                 round((acq.when - t).total_seconds() / 3600.0, 2), round(cover, 3), ref_day.isoformat(),
                                 len(refs))
                key = (abs(cand.offset_h), cand.pass_time)
                if best is None or key < (abs(best.offset_h), best.pass_time):
                    best = cand
        if best is not None:
            out.append(best)
    return out


def _product_extent(activation: Mapping[str, Any], product: Any) -> Any:
    for a in activation.get("aois", []):
        for p in a.get("products", []):
            if p.get("type") == product.product_type and int(p.get("monitoringNumber") or 0) == product.monitoring \
                    and int(p.get("aoiNumber") or a.get("number") or 0) == product.aoi:
                return p.get("extent")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#: Seconds between two uncached API requests: the public API rate-limits bursts (HTTP 403).
_API_PAUSE_S = 2.0


def cached_activations(codes: Sequence[str], cache_dir: Path, *, fetch: Any = None,
                       pause_s: float = _API_PAUSE_S, sleep: Any = None) -> list[dict[str, Any]]:
    """Activation records, each fetched once into ``cache_dir`` (``<code>.json``; ``<code>.missing`` if unknown).

    Uncached requests are spaced by ``pause_s``. An activation that is not public (HTTP 401/403)
    is recorded as ``<code>.restricted`` and skipped: the benchmark uses no credentials. Any other
    failure stops the sweep with what is cached so far preserved, so a re-run resumes there.
    """
    import time

    from floodfusion.data.cems import fetch_activation

    fetch = fetch or fetch_activation
    sleep = sleep or time.sleep
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for code in codes:
        hit, miss = cache_dir / f"{code}.json", cache_dir / f"{code}.missing"
        if hit.is_file():
            records.append(json.loads(hit.read_text(encoding="utf-8")))
            continue
        if miss.is_file() or (cache_dir / f"{code}.restricted").is_file():
            continue
        try:
            record = fetch(code)
        except LookupError:
            miss.write_text("", encoding="utf-8")
            sleep(pause_s)
            continue
        except OSError as exc:
            # Some activations are not public ("Authentication credentials were not provided"):
            # they are outside the benchmark by definition (no credentials), not an error.
            if not any(f"HTTP {s} " in f"{exc} " for s in (401, 403)):
                raise
            (cache_dir / f"{code}.restricted").write_text(str(exc)[:200], encoding="utf-8")
            sleep(pause_s)
            continue
        hit.write_text(json.dumps(record), encoding="utf-8")
        records.append(record)
        sleep(pause_s)
    return records


def _eligibility(args: argparse.Namespace) -> dict[str, Any]:
    from floodfusion.data.archive import IngestSources

    records = cached_activations([f"EMSR{n}" for n in range(args.first, args.last + 1)], args.cache)
    floods = [r for r in records if str(r.get("category", "")).lower() == "flood"]
    sources = IngestSources.open()
    selected: list[dict[str, Any]] = []
    rule = {"window_h": WINDOW_H, "min_coverage": MIN_COVERAGE, "reference_gap_days": REFERENCE_GAP_DAYS,
            "pre_event_window_days": PRE_EVENT_WINDOW_DAYS, "related_lookback_days": RELATED_LOOKBACK_DAYS,
            "requires_package": True}
    for record in floods:
        # Each activation's verdict is cached with the rule it was checked under: a resumed census
        # (after a catalogue refusal) skips what is done, and a changed rule re-checks everything.
        done = args.cache / f"{record['code']}.eligibility.json"
        cached = json.loads(done.read_text(encoding="utf-8")) if done.is_file() else None
        if cached is not None and cached.get("rule") == rule:
            pairs_dicts = cached["pairs"]
        else:
            pairs_dicts = [asdict(p) for p in eligible_pairs(record, sources.client, others=floods)]
            done.write_text(json.dumps({"rule": rule, "pairs": pairs_dicts}), encoding="utf-8")
        LOGGER.info("%s (%s, %s): %d eligible AOI(s)", record["code"], record.get("countries"), record.get("eventTime"),
                    len(pairs_dicts))
        selected += pairs_dicts
    summary = {
        "rule": {**rule, "codes": [f"EMSR{args.first}", f"EMSR{args.last}"]},
        "n_activations": len(records), "n_flood_activations": len(floods),
        "eligible_activations": sorted({p["activation"] for p in selected}),
        "pairs": selected,
        "checked": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("%d of %d flood activations eligible -> %s", len(summary["eligible_activations"]), len(floods), args.out)
    return summary


#: Sentinel-2 is searched this many days either side of the pass (real clouds, operational).
OPTICAL_WINDOW_DAYS = 2
#: The activation is the unit, so a huge one need not be mapped whole: at most this many AOIs
#: (smallest |time offset| first, ties by AOI number) ...
MAX_AOIS_PER_ACTIVATION = 4
#: ... and at most this many ~16 x 17 km tiles per AOI, drawn with a generator seeded by the
#: pair's key (never by content: the sample is blind to the flood and to every model).
MAX_TILES_PER_AOI = 16


def select_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """At most :data:`MAX_AOIS_PER_ACTIVATION` pairs per activation, the closest in time."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for pair in pairs:
        by.setdefault(str(pair["activation"]), []).append(pair)
    out: list[Mapping[str, Any]] = []
    for activation in sorted(by):
        out += sorted(by[activation], key=lambda p: (abs(float(p["offset_h"])), int(p["aoi"])))[:MAX_AOIS_PER_ACTIVATION]
    return out


def sample_tiles(tiles: Sequence[tuple[str, Any]], key: str) -> list[tuple[str, Any]]:
    """At most :data:`MAX_TILES_PER_AOI` of ``tiles``, by a generator seeded with ``key``; in tile order."""
    import hashlib

    import numpy as np

    if len(tiles) <= MAX_TILES_PER_AOI:
        return list(tiles)
    rng = np.random.default_rng(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16))
    keep = sorted(rng.permutation(len(tiles))[:MAX_TILES_PER_AOI])
    return [tiles[i] for i in keep]


def pair_key(pair: Mapping[str, Any]) -> str:
    """``EMSR664_AOI01``: one eligible pair per AOI, so the AOI names it."""
    return f"{pair['activation']}_AOI{int(pair['aoi']):02d}"


def prepare_pair(pair: Mapping[str, Any], record: Mapping[str, Any], out_root: Path, cfg: Any,
                 sources: Any) -> dict[str, Any]:
    """Download the pair's product, write its label and scored region, ingest its pass tile by tile.

    Resumable: a tile whose ``event.json`` exists is not ingested again; ``pair.json`` is
    rewritten with every tile's outcome.
    """
    from floodfusion.case_study import _intersects, _write_geojson, scored_region, tiles_for
    from floodfusion.data import cems
    from floodfusion.data.ingest import EventConfig, ingest_event
    from floodfusion.data.stac import NoMatchingPairError, redact

    [product] = [p for p in cems.list_products(record) if p.tag == pair["product"]]
    out = out_root / pair_key(pair)
    product_dir = cems.download_product(product, out_root / "cems")
    region = scored_region(product_dir)
    flood = cems.observed_flood(product_dir)
    label = _write_geojson(out / "labels" / "flood.geojson", list(flood.geometry))
    region_path = _write_geojson(out / "labels" / "region.geojson", [region])
    when = _time(pair["pass_time"]).date()
    tiles = sample_tiles([(t, b) for t, b in tiles_for(region.bounds) if _intersects(region, b)], pair_key(pair))
    status: dict[str, str] = {}
    for tile_id, bbox in tiles:
        event = EventConfig(event_id=f"{pair_key(pair)}_{tile_id}", bbox=tuple(bbox), flood_start=when, flood_end=when,
                            label_path=label, aoi_path=region_path, relative_orbit=int(pair["relative_orbit"]),
                            orbit_state=str(pair["orbit_state"]),
                            reference_before=date.fromisoformat(str(pair["reference_before"])),
                            optical_window_days=OPTICAL_WINDOW_DAYS)
        if (out / "events" / event.event_id / "event.json").is_file():
            status[tile_id] = "ok"
            continue
        try:
            ingest_event(event, out, cfg, sources=sources)
            status[tile_id] = "ok"
        except NoMatchingPairError as exc:
            status[tile_id] = f"dropped: {exc}"
        except Exception as exc:  # noqa: BLE001 - one tile must not abort the pair
            status[tile_id] = f"failed: {type(exc).__name__}: {redact(str(exc))[:300]}"
        LOGGER.info("%s %s: %s", pair_key(pair), tile_id, status[tile_id][:120])
    summary = {**dict(pair), "key": pair_key(pair), "tiles": status, "label": str(label), "region": str(region_path),
               "flood_polygons": int(len(flood))}
    (out / "pair.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _prepare(args: argparse.Namespace) -> None:
    from floodfusion.config import load_config
    from floodfusion.data.archive import IngestSources

    selection = json.loads(args.pairs.read_text(encoding="utf-8"))
    cfg = load_config(args.config, overlays=args.overlay)
    sources = IngestSources.open()
    wanted = set(args.activation or [])
    for pair in select_pairs(selection["pairs"]):
        if wanted and pair["activation"] not in wanted:
            continue
        record = json.loads((args.cache / f"{pair['activation']}.json").read_text(encoding="utf-8"))
        try:
            summary = prepare_pair(pair, record, args.out, cfg, sources)
        except Exception as exc:  # noqa: BLE001 - one pair's failure must not stop the benchmark
            from floodfusion.data.http import redacted_traceback

            LOGGER.error("%s failed: %s\n%s", pair_key(pair), exc, redacted_traceback(exc))
            continue
        ok = sum(v == "ok" for v in summary["tiles"].values())
        LOGGER.info("%s: %d/%d tiles ingested", summary["key"], ok, len(summary["tiles"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CEMS flood delineations as an independent benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    elig = sub.add_parser("eligibility", help="apply the selection rule to an activation code range (metadata only)")
    elig.add_argument("--first", type=int, default=656)
    elig.add_argument("--last", type=int, default=898)
    elig.add_argument("--out", type=Path, default=Path("configs/datasets/cems_benchmark.json"))
    elig.add_argument("--cache", type=Path, default=Path("data/cems_records"))
    prep = sub.add_parser("prepare", help="download each eligible pair's product and ingest its pass")
    prep.add_argument("--pairs", type=Path, default=Path("configs/datasets/cems_benchmark.json"))
    prep.add_argument("--cache", type=Path, default=Path("data/cems_records"))
    prep.add_argument("--out", type=Path, default=Path("data/cems_benchmark"))
    prep.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    prep.add_argument("--overlay", type=Path, action="append", default=[])
    prep.add_argument("--activation", action="append", default=None, help="only this activation (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "eligibility":
        _eligibility(args)
    else:
        _prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
