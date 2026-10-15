"""Erratum 2's exposure: how much of each dataset's stack area lay past the first DEM tile a stack lists.

    uv run --frozen python scripts/dem_exposure.py      # -> reports/v1_1_erratum2_exposure.md

Before the fix, a stack whose grid needed two GLO-30 tiles read every pixel outside the **first** tile as terrain
0 m (`reports/v1_1_erratum.md`, erratum 2). For each v1.1 stack this script takes the grid's footprint and the first
tile's 1° cell (from the tile id in ``event.json``'s ``dem_ids``), and measures the share of the grid outside that
cell, in the stack's own projection. That share is what the defect zeroed.

Per dataset it reports the stacks affected, the median share among them, and the total in whole-stack equivalents
and as a share of all stack area. It reads metadata only (``event.json``); no raster is opened.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (("Sen1Floods11", "data/sen1floods11/events/*/event.json"),
            ("Kuro Siwo", "data/kurosiwo/events/*/event.json"),
            ("CEMS benchmark", "data/cems_benchmark/EMSR*_AOI*/events/*/event.json"))
_TILE = re.compile(r"_(?P<ns>[NS])(?P<lat>\d{2})_00_(?P<ew>[EW])(?P<lon>\d{3})_00_")
#: A share below this is a rounding sliver of a grid that sits on a tile edge, not exposure.
MIN_SHARE = 1e-4


def first_tile_cell(dem_ids: object) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` of the 1° cell of the first GLO-30 tile id listed."""
    ids = json.loads(dem_ids.replace("'", '"')) if isinstance(dem_ids, str) else list(dem_ids)
    m = _TILE.search(ids[0])
    if m is None:
        raise ValueError(f"not a GLO-30 tile id: {ids[0]!r}")
    lat = int(m["lat"]) * (1 if m["ns"] == "N" else -1)
    lon = int(m["lon"]) * (1 if m["ew"] == "E" else -1)
    return float(lon), float(lat), float(lon + 1), float(lat + 1)


def outside_share(meta: dict) -> float:
    """Share of the stack's grid outside the first DEM tile's cell (0 for a stack within one tile)."""
    import numpy as np
    import shapely
    from pyproj import Transformer
    from shapely.geometry import box

    t = meta["transform"] if isinstance(meta["transform"], list) else json.loads(meta["transform"])
    w, h = int(meta["width"]), int(meta["height"])
    grid = box(t[2], t[5] + t[4] * h, t[2] + t[0] * w, t[5])
    to_grid = Transformer.from_crs(4326, int(meta["epsg"]), always_xy=True)
    cell = shapely.transform(box(*first_tile_cell(meta["dem_ids"])).segmentize(0.01),
                             lambda xy: np.column_stack(to_grid.transform(xy[:, 0], xy[:, 1])))
    share = 1.0 - grid.intersection(cell).area / grid.area
    return share if share > MIN_SHARE else 0.0


def exposure(pattern: str) -> dict[str, float]:
    shares, areas = [], []
    for path in sorted(ROOT.glob(pattern)):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not meta.get("dem_ids"):
            continue
        t = meta["transform"] if isinstance(meta["transform"], list) else json.loads(meta["transform"])
        shares.append(outside_share(meta))
        areas.append(abs(t[0] * t[4]) * int(meta["width"]) * int(meta["height"]))
    hit = [s for s in shares if s > 0]
    return {"stacks": len(shares), "affected": len(hit), "median_share": statistics.median(hit) if hit else 0.0,
            "whole_stacks": sum(hit), "area_share": sum(s * a for s, a in zip(shares, areas)) / sum(areas)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v1_1_erratum2_exposure.md")
    args = parser.parse_args(argv)
    lines = ["# Erratum 2's exposure: stack area past the first DEM tile (v1.1 data)", "",
             "Written by `scripts/dem_exposure.py` from each stack's `event.json` (metadata only). A stack is affected "
             "when part of its grid lies outside the 1° cell of the first GLO-30 tile it lists: before the fix, that "
             "part read terrain 0 m.", "",
             "| Dataset | Stacks affected | Median share of an affected stack | Total, in whole stacks | Share of all stack area |",
             "|---|---|---|---|---|"]
    for name, pattern in DATASETS:
        e = exposure(pattern)
        lines.append(f"| {name} | {e['affected']} of {e['stacks']} | {e['median_share']:.2f} | "
                     f"≈ {e['whole_stacks']:.0f} | {100 * e['area_share']:.1f} % |")
        print(name, e)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
