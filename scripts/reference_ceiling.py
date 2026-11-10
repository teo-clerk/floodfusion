r"""The reference ceiling: how well do two Copernicus EMS maps of the same area, made at most 24 h apart, agree?

Every model in this repository scores about 0.42 IoU against Copernicus EMS delineations. Those delineations are
products, not ground truth: an analyst maps a commercial image, hours from the Sentinel-1 pass. This script measures
how well the reference agrees with itself, under rules fixed before it ran (``reports/protocol_v1_3.md`` §2.3, §5):

- **Set:** seen flood activations only (EMSR656–EMSR898; the blind set starts at EMSR899 and is refused). For every
  AOI, consecutive delineation (DEL) products, ordered by their latest source image, at most 24 h apart.
- **Comparison:** each product's observed flood, on a 10 m UTM grid, inside the intersection of the two products'
  observed regions (:func:`floodfusion.case_study.scored_region`).
- **Metric:** IoU per product pair; per activation, the IoU of its pooled counts; the macro mean over activations.
  Strata: the gap between the two maps, the sensor pair, and whether they share a source image.

Two steps, so that the data can be checked before any agreement is computed::

    uv run --frozen python scripts/reference_ceiling.py download   # products (network) + the carry-over probe
    uv run --frozen python scripts/reference_ceiling.py score      # counts per pair (CPU) + the report

**The carry-over probe.** A monitoring product may repeat flood polygons drawn on an earlier image. Two maps that
share polygons agree by construction, so ``download`` reports, per pair, the share of the later product's flood area
drawn on images no later than the earlier product's latest image.

Contains modified Copernicus Emergency Management Service information (2023–2026).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.cems_benchmark import sensor_kind
from floodfusion.data.cems import CemsProduct, list_products

LOGGER = logging.getLogger("reference_ceiling")

#: Seen activations only: the blind set (protocol v1.2 §2.2) starts at EMSR899.
FIRST, LAST = 656, 898
MAX_GAP_H = 24.0
PIXEL_M = 10.0
SHORT_GAP_H = 6.0
_KINDS = ("radar", "optical")
_CODE = re.compile(r"EMSR([0-9]{3,4})", re.ASCII)


# ---------------------------------------------------------------------------
# Candidates (metadata only)
# ---------------------------------------------------------------------------
def image_kind(sensor: str) -> str:
    """``radar``, ``optical`` or ``unknown`` for one sensor name (Sentinel-1 is radar, Sentinel-2 optical)."""
    name = str(sensor or "").strip()
    if name.lower().startswith("sentinel"):
        number = name.lower().removeprefix("sentinel").lstrip("- ")
        return "radar" if number.startswith("1") else "optical" if number.startswith("2") else "unknown"
    return {"sar": "radar", "optical": "optical"}.get(sensor_kind(name), "unknown")


def product_kind(product: CemsProduct) -> str:
    kinds = {image_kind(sensor) for sensor, _ in product.images}
    if not kinds:
        return "unknown"
    return kinds.pop() if len(kinds) == 1 else "mixed"


def sensor_pair(first: str, second: str) -> str:
    """``radar–radar``, ``radar–optical`` or ``optical–optical``; ``other`` if either is mixed or unknown."""
    if first not in _KINDS or second not in _KINDS:
        return "other"
    return "–".join(sorted((first, second), key=_KINDS.index))


@dataclass(frozen=True)
class Candidate:
    activation: str
    aoi: int
    first: CemsProduct
    second: CemsProduct
    gap_h: float

    @property
    def key(self) -> str:
        parts = (p.tag.split(f"_{p.product_type}_", 1)[1] for p in (self.first, self.second))
        return f"{self.activation}_AOI{self.aoi:02d}_" + "_".join(parts)

    @property
    def sensor_pair(self) -> str:
        return sensor_pair(product_kind(self.first), product_kind(self.second))

    @property
    def shared_image(self) -> bool:
        return bool(set(self.first.images) & set(self.second.images))


def _seen_code(code: str) -> int:
    match = _CODE.fullmatch(str(code))
    if match is None:
        raise ValueError(f"not an activation code: {str(code)[:20]!r}")
    number = int(match.group(1))
    if not FIRST <= number <= LAST:
        raise ValueError(f"{code} is outside the seen activations EMSR{FIRST}–EMSR{LAST} "
                         f"(the blind set starts at EMSR{LAST + 1})")
    return number


def candidates(records: Iterable[Mapping[str, Any]]) -> list[Candidate]:
    """Consecutive delineations of one AOI, at most :data:`MAX_GAP_H` apart, of every seen flood activation."""
    out = []
    for record in records:
        _seen_code(record.get("code", ""))
        if str(record.get("category", "")).lower() != "flood":
            continue
        by_aoi: dict[int, list[CemsProduct]] = defaultdict(list)
        for product in list_products(record):
            if product.acquisition_times and product.download_url:
                by_aoi[product.aoi].append(product)
        for aoi, products in sorted(by_aoi.items()):
            ordered = sorted(products, key=lambda p: (max(p.acquisition_times), p.monitoring))
            for first, second in pairwise(ordered):
                gap = (max(second.acquisition_times) - max(first.acquisition_times)).total_seconds() / 3600.0
                if gap <= MAX_GAP_H:
                    out.append(Candidate(first.activation, aoi, first, second, gap))
    return out


def load_records(directory: Path) -> list[dict[str, Any]]:
    """The cached records of the seen codes only; a blind code's file is never opened."""
    paths = (directory / f"EMSR{n}.json" for n in range(FIRST, LAST + 1))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths if p.is_file()]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def _mask(geometries: Sequence[Any], grid: Any) -> np.ndarray:
    """EPSG:4326 polygons on the grid, pixel centres inside (as ``ingest.rasterise_vector``)."""
    import geopandas as gpd
    from rasterio.features import rasterize

    shapes = [g for g in gpd.GeoSeries(list(geometries), crs=4326).to_crs(grid.epsg) if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((grid.height, grid.width), dtype=bool)
    return rasterize(((g, 1) for g in shapes), out_shape=(grid.height, grid.width), transform=grid.affine, fill=0,
                     all_touched=False, dtype="uint8").astype(bool)


def counts(region_first: Any, region_second: Any, flood_first: Sequence[Any], flood_second: Sequence[Any], *,
           pixel_m: float = PIXEL_M) -> dict[str, int] | None:
    """Pixel counts of the two maps on the region both observed; None if they observed no common area."""
    from floodfusion.data.chipping import build_analysis_grid

    common = region_first.intersection(region_second)
    if common.is_empty or common.area == 0:
        return None
    grid = build_analysis_grid(tuple(common.bounds), pixel_m)
    region = _mask([common], grid)
    a, b = _mask(flood_first, grid) & region, _mask(flood_second, grid) & region
    return {"region_px": int(region.sum()), "both": int((a & b).sum()), "only_first": int((a & ~b).sum()),
            "only_second": int((~a & b).sum())}


def iou(c: Mapping[str, int]) -> float:
    union = c["both"] + c["only_first"] + c["only_second"]
    return c["both"] / union if union else math.nan


def carried_over_share(flood: Sequence[tuple[Any, datetime | None]], *, first_latest: datetime) -> float:
    """Share of a later product's flood area drawn on images no later than the earlier product's latest image."""
    total = sum(g.area for g, _ in flood)
    old = sum(g.area for g, when in flood if when is not None and when <= first_latest)
    return old / total if total else 0.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _block(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"both": 0, "only_first": 0, "only_second": 0})
    for p in pairs:
        for k in pooled[p["activation"]]:
            pooled[p["activation"]][k] += p["counts"][k]
    per = {a: iou(c) for a, c in sorted(pooled.items())}
    defined = [v for v in per.values() if not math.isnan(v)]
    pair_ious = [v for v in (iou(p["counts"]) for p in pairs) if not math.isnan(v)]
    return {"activations": len(defined), "undefined_activations": len(per) - len(defined), "pairs": len(pairs),
            "macro_iou": statistics.fmean(defined) if defined else math.nan,
            "median_pair_iou": statistics.median(pair_ious) if pair_ious else math.nan,
            "per_activation": per}


def _strata(pairs: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for p in pairs:
        groups[key(p)].append(p)
    return {k: _block(v) for k, v in sorted(groups.items())}


def summarise(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"all": _block(pairs),
            "gap": _strata(pairs, lambda p: f"< {SHORT_GAP_H:g} h" if p["gap_h"] < SHORT_GAP_H
                           else f"{SHORT_GAP_H:g}–{MAX_GAP_H:g} h"),
            "sensor_pair": _strata(pairs, lambda p: p["sensor_pair"]),
            "shared_image": _strata(pairs, lambda p: "shared" if p["shared_image"] else "not shared")}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _f(x: float) -> str:
    return "—" if x is None or math.isnan(x) else f"{x:.3f}"


def _rows(blocks: Mapping[str, Mapping[str, Any]], label: str) -> list[str]:
    lines = [f"| {label} | Activations | Product pairs | Macro IoU over activations | Median pair IoU |",
             "| --- | --- | --- | --- | --- |"]
    for name, b in blocks.items():
        lines.append(f"| {name} | {b['activations']} | {b['pairs']} | {_f(b['macro_iou'])} | {_f(b['median_pair_iou'])} |")
    return lines


def render(summary: Mapping[str, Any], skipped: Mapping[str, str], probe: Mapping[str, float]) -> str:
    shares = sorted(probe.values())
    lines = [
        "# The reference ceiling: two Copernicus EMS maps of the same area, at most 24 h apart", "",
        "Seen flood activations EMSR656–EMSR898 only; descriptive, no test. Rules fixed before this was computed "
        "(`reports/protocol_v1_3.md` §2.3, §5). For each AOI, consecutive delineation products whose latest source "
        "images are at most 24 h apart are compared on the area both observed (10 m UTM grid; each product's observed "
        "flood). An activation pools its pairs' pixels; activations where neither map holds flood are left out.", "",
        "**Reading, fixed in the protocol:** set against the models' 0.42 on radar-drawn pairs. Agreement near 0.5 "
        "means the benchmark is near its ceiling; near 0.8, most of the gap is the models'. Two maps hours apart also "
        "differ because the flood moved, as a model's map and its reference do.", "",
        *_rows({"All pairs": summary["all"]}, ""), "",
        "## By time between the two maps", "", *_rows(summary["gap"], "Gap"), "",
        "## By sensor pair", "", *_rows(summary["sensor_pair"], "Sensors"), "",
        "## By shared source image", "", *_rows(summary["shared_image"], "Source image"), "",
        "## Carried-over flood (the probe)", "",
        f"Share of the later product's flood area drawn on images no later than the earlier product's latest image, "
        f"over {len(shares)} pairs: median {_f(statistics.median(shares)) if shares else '—'}, "
        f"{sum(s > 0.05 for s in shares)} pairs above 5 %.", "",
        "## Per activation", "", "| Activation | IoU (pairs pooled) |", "| --- | --- |",
        *(f"| {a} | {_f(v)} |" for a, v in summary["all"]["per_activation"].items()), "",
    ]
    if skipped:
        lines += ["## Pairs not compared", "", "| Pair | Reason |", "| --- | --- |",
                  *(f"| {k} | {v} |" for k, v in sorted(skipped.items())), ""]
    lines += ["*Regenerate:* `uv run --frozen python scripts/reference_ceiling.py download` then `… score`.", "",
              "Contains modified Copernicus Emergency Management Service information (2023–2026).", ""]
    return "\n".join(lines)


def _nan_to_none(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _download(found: Sequence[Candidate], products: Path, work: Path) -> None:
    """Fetch both products of every pair (resumable); a product the API no longer serves skips its pair only."""
    import zipfile

    from floodfusion.data.cems import download_product, observed_flood
    from floodfusion.data.http import HTTPStatusError, redact

    probe, failed = {}, {}
    for c in found:
        try:
            dirs = [download_product(p, products) for p in (c.first, c.second)]
        except (HTTPStatusError, OSError, ValueError, zipfile.BadZipFile) as exc:
            failed[c.key] = f"download failed: {type(exc).__name__}: {redact(str(exc))[:160]}"
            LOGGER.warning("%s: %s", c.key, failed[c.key])
            continue
        flood = observed_flood(dirs[1])
        probe[c.key] = carried_over_share(list(zip(flood.geometry, flood["source_time"], strict=True)),
                                          first_latest=max(c.first.acquisition_times))
    work.mkdir(parents=True, exist_ok=True)
    (work / "probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True), encoding="utf-8")
    (work / "download_failures.json").write_text(json.dumps(failed, indent=2, sort_keys=True), encoding="utf-8")
    above = sum(v > 0.05 for v in probe.values())
    LOGGER.info("%d pairs downloaded, %d failed; carried-over share above 5 %% in %d", len(probe), len(failed), above)


def _score(found: Sequence[Candidate], products: Path, work: Path, out: Path) -> None:
    from floodfusion.case_study import scored_region
    from floodfusion.data.cems import observed_flood

    failures = work / "download_failures.json"
    pairs, skipped = [], dict(json.loads(failures.read_text(encoding="utf-8")) if failures.is_file() else {})
    for c in found:
        if c.key in skipped:
            continue
        cached = work / "pairs" / f"{c.key}.json"
        if not cached.is_file():
            dirs = [products / c.activation / f"AOI{c.aoi:02d}" / p.tag for p in (c.first, c.second)]
            try:
                regions = [scored_region(d) for d in dirs]
                result = counts(*regions, *(list(observed_flood(d).geometry) for d in dirs))
                entry = {"reason": "no common observed region"} if result is None else {"counts": result}
            except (FileNotFoundError, ValueError) as exc:
                entry = {"reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
        entry = json.loads(cached.read_text(encoding="utf-8"))
        if "counts" not in entry:
            skipped[c.key] = entry["reason"]
            continue
        pairs.append({"key": c.key, "activation": c.activation, "gap_h": round(c.gap_h, 2),
                      "sensor_pair": c.sensor_pair, "shared_image": c.shared_image, "counts": entry["counts"]})
    summary = summarise(pairs)
    probe = json.loads((work / "probe.json").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(_nan_to_none(
        {"rule": {"first": FIRST, "last": LAST, "max_gap_h": MAX_GAP_H, "pixel_m": PIXEL_M},
         "summary": summary, "pairs": pairs, "skipped": skipped, "carried_over_share": probe}),
        indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    out.with_suffix(".md").write_text(render(summary, skipped, probe), encoding="utf-8")
    LOGGER.info("macro IoU over %d activations: %s", summary["all"]["activations"], _f(summary["all"]["macro_iou"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("step", choices=("download", "score"))
    parser.add_argument("--records", type=Path, default=Path("data/cems_records"))
    parser.add_argument("--products", type=Path, default=Path("data/reference_ceiling/cems"))
    parser.add_argument("--work", type=Path, default=Path("data/reference_ceiling"))
    parser.add_argument("--out", type=Path, default=Path("reports/reference_ceiling.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    found = candidates(load_records(args.records))
    LOGGER.info("%d candidate pairs in %d activations", len(found), len({c.activation for c in found}))
    if args.step == "download":
        _download(found, args.products, args.work)
    else:
        _score(found, args.products, args.work, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
