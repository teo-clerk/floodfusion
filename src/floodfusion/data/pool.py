r"""The v1.2 training pool: several re-ingested datasets as one chip manifest, regrouped jointly by place.

No raster is copied. The pooled manifest lists the chips of every source root (Kuro Siwo rebuilt, GEOID-Flood), with
raster paths relative to its own folder, so :mod:`floodfusion.crossval` reads it like any dataset root.

**What it changes.** Only the split group (``basin_id``).
- Groups are recomputed over all source events together: events whose footprints lie within 100 km form one group
  (single linkage, :func:`floodfusion.data.kurosiwo.event_groups`).
- A region flooded twice, once in each dataset, can therefore never train a model and test it.
- A source event is the dataset's own event (``kurosiwo.event`` or ``geoid_flood.event`` in each stack's metadata).
  Its footprint is the union of its stacks' grids.

**The size rule (fixed before any run).** If the largest group holds more than :data:`MAX_GROUP_SHARE` of the
windows, :func:`build_pool` refuses. Cross-validation would then need buffered spatial blocks.

**v1.3: the CEMS benchmark as a training source** (``reports/protocol_v1_3.md`` §2.1). Its pairs are separate dataset
roots without a manifest. :func:`collect_cems` lists every pair's tiles as one root, and :func:`build_pool` then
regroups it like any other. A CEMS stack's event is its activation, read from the stack id
(``EMSR657_AOI01_t00_00`` → ``cems:EMSR657``), so all AOIs of one activation share a group.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

__all__ = ["BLIND_FIRST_CODE", "GROUP_PREFIX", "MAX_GROUP_SHARE", "build_pool", "collect_cems", "main",
           "source_event"]

LOGGER = logging.getLogger(__name__)

GROUP_PREFIX = "POOL_"
MAX_GROUP_SHARE = 0.25
#: The stack metadata blocks that name a dataset's own event, per source.
_SOURCE_KEYS = ("kurosiwo", "geoid_flood")
#: A CEMS benchmark stack id: ``<activation>_AOI<nn>_<tile>``.
_CEMS_STACK = re.compile(r"(EMSR[0-9]{3,4})_AOI[0-9]{2}_.+", re.ASCII)
#: The blind test's first activation (``protocol_v1_2.md`` §2.2): never a training source.
BLIND_FIRST_CODE = 899


def source_event(stack_meta: dict[str, Any]) -> str:
    """The dataset's own event a stack belongs to, prefixed by its source (``kurosiwo:EMSR118_Spain``)."""
    for key in _SOURCE_KEYS:
        block = stack_meta.get(key)
        if isinstance(block, dict) and block.get("event"):
            return f"{key}:{block['event']}"
    match = _CEMS_STACK.fullmatch(str(stack_meta.get("event_id", "")))
    if match:
        return f"cems:{match.group(1)}"
    raise KeyError(f"stack {stack_meta.get('event_id')!r} names no source event ({', '.join(_SOURCE_KEYS)}, "
                   "or a CEMS stack id)")


def collect_cems(benchmark: Path, out: Path, *, exclude: Collection[str] = ("EMSR664",),
                 manifest_name: str = "manifest.parquet") -> str:
    """Write ``out/chips/<manifest_name>`` listing the chips of every benchmark pair; return its hash.

    A pair is a folder holding ``pair.json``; activations in ``exclude`` are left out (EMSR664, Forlì, is v1.0's
    case study). Its record names each pair's stratum (``provenance.strata``). An activation of the blind set refuses
    the whole collection.
    """
    from floodfusion.data.records import load_manifest, write_manifest

    records, strata = [], {}
    for pair_file in sorted(Path(benchmark).glob("*/pair.json")):
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
        activation = str(pair["activation"])
        if int(activation.removeprefix("EMSR")) >= BLIND_FIRST_CODE:
            raise ValueError(f"{pair['key']}: {activation} belongs to the blind set, never to training")
        if activation in exclude:
            continue
        strata[str(pair["key"])] = str(pair["stratum"])
        for chips in sorted((pair_file.parent / "events").glob("*/chips.parquet")):
            records.extend(load_manifest(chips))
    return write_manifest(records, Path(out) / "chips" / manifest_name, provenance={
        "generator": "floodfusion.data.pool.collect_cems", "benchmark": str(benchmark), "strata": strata,
        "excluded": sorted(exclude)})


def _footprint(meta: dict[str, Any]) -> Any:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    t = meta["transform"] if isinstance(meta["transform"], list) else json.loads(meta["transform"])
    w, h = int(meta["width"]), int(meta["height"])
    bounds = (t[2], t[5] + t[4] * h, t[2] + t[0] * w, t[5])
    return box(*transform_bounds(CRS.from_epsg(int(meta["epsg"])), CRS.from_epsg(4326), *bounds))


def build_pool(roots: Sequence[Path], out: Path, *, manifest_name: str = "manifest.parquet") -> dict[str, Any]:
    """Write ``out/chips/<manifest_name>`` pooling every root's chips, regrouped jointly; return a summary."""
    from shapely.ops import unary_union

    from floodfusion.data.kurosiwo import event_groups
    from floodfusion.data.records import load_manifest, load_manifest_meta, write_manifest

    records, event_of_stack, shapes, sources = [], {}, {}, {}
    for root in roots:
        manifest = Path(root) / "chips" / manifest_name
        sources[str(root)] = load_manifest_meta(manifest)["manifest_sha256"]
        for record in load_manifest(manifest):
            stack = Path(record.raster_path)
            if stack not in event_of_stack:
                meta = json.loads((stack.parent / "event.json").read_text(encoding="utf-8"))
                event_of_stack[stack] = source_event(meta)
                shapes.setdefault(event_of_stack[stack], []).append(_footprint(meta))
            records.append(record)
    groups = event_groups({e: unary_union(s) for e, s in shapes.items()})
    group_of = {e: GROUP_PREFIX + g.split("_", 1)[1].replace(":", "_") for e, g in groups.items()}
    pooled = [replace(r, basin_id=group_of[event_of_stack[Path(r.raster_path)]]) for r in records]
    windows = Counter(r.basin_id for r in pooled)
    largest, n_largest = windows.most_common(1)[0]
    share = n_largest / len(pooled)
    if share > MAX_GROUP_SHARE:
        raise ValueError(f"group {largest} holds {share:.0%} of the windows (> {MAX_GROUP_SHARE:.0%}): "
                         "use buffered spatial blocks instead (reports/v1_2_deviations.md)")
    summary = {"sources": sources, "n_chips": len(pooled), "n_events": len(shapes), "n_groups": len(windows),
               "largest_group": largest, "largest_share": round(share, 4),
               "group_of_event": dict(sorted(group_of.items()))}
    digest = write_manifest(pooled, Path(out) / "chips" / manifest_name, provenance=summary)
    return {**summary, "manifest_sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool re-ingested datasets into one manifest, regrouped by place.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, nargs="+", help="dataset roots with chips/manifest.parquet")
    source.add_argument("--cems", type=Path, help="a CEMS benchmark folder: list its pairs as one root at --out")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cems is not None:
        LOGGER.info("CEMS root %s: manifest %s", args.out, collect_cems(args.cems, args.out)[:12])
        return 0
    summary = build_pool(args.root, args.out)
    LOGGER.info("%d chips, %d events, %d groups; largest %s (%.1f %%); manifest %s", summary["n_chips"],
                summary["n_events"], summary["n_groups"], summary["largest_group"], 100 * summary["largest_share"],
                summary["manifest_sha256"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
