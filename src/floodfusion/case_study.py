r"""A real activation, end to end: every Sentinel-1 pass of a flood against CEMS's own maps.

Run as (``make forli``)::

    python -m floodfusion.case_study prepare --activation EMSR664 --aoi 1 \
        --start 2023-05-16 --end 2023-05-23 --reference-before 2023-05-01 --out data/emsr664
    python -m floodfusion.case_study run --case data/emsr664 \
        --fusion-run runs/sen1floods11/official/official/row09_fusion_crossattn/seed42 \
        --sar-only-run runs/sen1floods11/official/official/row11_fusion_eval_sar_only/seed42 \
        --baseline-run runs/sen1floods11/official/official/row02_baseline_adaptive/baseline \
        --out reports/emsr664_forli.md

``prepare`` downloads the activation's delineation products for one AOI, finds every
Sentinel-1 acquisition over it in the window, and ingests each **pass x tile** (the AOI is
tiled so a 75 x 66 km area fits a laptop's memory) with the pre-event reference forced
before ``--reference-before``. ``run`` predicts every tile with the model (trained on
Sen1Floods11 only — this event is **zero-shot**), the same model SAR-only, and the physics
baseline, and scores each pass against **every** CEMS product on the area that product
actually observed (its AOI inside its post-event image footprints, minus "not analysed").

The pass x product matrix is the point: agreement should peak at the product nearest in
time and fall as the flood recedes. A Sentinel-1 map compared with a delineation taken 12
hours earlier is a comparison of two instants, and a 3 m X-band delineation and a 10 m
C-band map do not share a minimum mapping unit — both are reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import ConfusionMatrix

__all__ = ["Pass", "find_passes", "main", "scored_region", "tiles_for"]

LOGGER = logging.getLogger(__name__)

#: ~16 x 17 km tiles at 44 N: ~2.7 Mpx each, ~0.6 GB peak during ingest.
TILE_DEG = (0.2, 0.15)
_ACQUISITION_GAP_S = 180


@dataclass(frozen=True)
class Pass:
    when: datetime
    relative_orbit: int
    orbit_state: str

    @property
    def tag(self) -> str:
        return f"S1_{self.when:%Y%m%dT%H%M}_o{self.relative_orbit:03d}"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def find_passes(client: Any, geometry: Any, start: date, end: date) -> list[Pass]:
    """Every distinct Sentinel-1 RTC acquisition over ``geometry`` (frames of one pass merged)."""
    from shapely.geometry import shape

    from floodfusion.data.stac import _datetime_of

    result = client.search(collections=["sentinel-1-rtc"], bbox=list(geometry.bounds),
                           datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z", max_items=200)
    items = [i if isinstance(i, dict) else i.to_dict() for i in (result.items() if hasattr(result, "items") else result)]
    items = [i for i in items if i.get("geometry") is None or shape(i["geometry"]).intersects(geometry)]
    passes: list[Pass] = []
    for item in sorted(items, key=_datetime_of):
        when = _datetime_of(item)
        p = item["properties"]
        candidate = Pass(when, int(p["sat:relative_orbit"]), str(p["sat:orbit_state"]).lower())
        if passes and passes[-1].relative_orbit == candidate.relative_orbit and \
                (when - passes[-1].when).total_seconds() <= _ACQUISITION_GAP_S:
            continue  # the next frame of the same acquisition
        passes.append(candidate)
    return passes


def tiles_for(bounds: Sequence[float], tile_deg: Sequence[float] = TILE_DEG) -> list[tuple[str, tuple[float, ...]]]:
    """Regular lon/lat tiles covering ``bounds``; ids ``t<row><col>`` from the north-west."""
    west, south, east, north = bounds
    # (12.1 - 11.7) / 0.2 is 2.0000000000000018 in binary: without the tolerance a sliver
    # column of empty tiles appears.
    nx = math.ceil((east - west) / tile_deg[0] - 1e-9)
    ny = math.ceil((north - south) / tile_deg[1] - 1e-9)
    out = []
    for r in range(ny):
        for c in range(nx):
            x0, y1 = west + c * tile_deg[0], north - r * tile_deg[1]
            out.append((f"t{r}{c}", (x0, max(south, y1 - tile_deg[1]), min(east, x0 + tile_deg[0]), y1)))
    return out


def scored_region(product_dir: Path) -> Any:
    """Where a product observed: its AOI inside its post-event image footprints, minus 'Not Analysed'."""
    import geopandas as gpd

    from floodfusion.data.cems import _layer, aoi_geometry, source_times

    aoi = aoi_geometry(product_dir)
    path = _layer(product_dir, "imageFootprintA")
    if path is None:
        return aoi
    frame = gpd.read_file(path)
    frame = frame.set_crs(4326) if frame.crs is None else frame.to_crs(4326)
    post = set(source_times(product_dir))
    kind = frame.get("obj_type", gpd.pd.Series([""] * len(frame))).astype(str)
    src = frame.get("or_src_id", gpd.pd.Series([None] * len(frame)))
    seen = frame[(kind == "Image Footprint") & src.isin(post)]
    blind = frame[kind.str.contains("Not Analysed", case=False)]
    region = aoi.intersection(seen.geometry.union_all()) if len(seen) else aoi
    return region.difference(blind.geometry.union_all()) if len(blind) else region


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def _write_geojson(path: Path, geometries: Sequence[Any], properties: Sequence[Mapping[str, Any]] | None = None) -> Path:
    import geopandas as gpd

    frame = gpd.GeoDataFrame(list(properties) if properties else [{}] * len(geometries),
                             geometry=list(geometries), crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    return path


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from floodfusion.config import load_config
    from floodfusion.data import cems
    from floodfusion.data.ingest import EventConfig, IngestSources, ingest_event
    from floodfusion.data.stac import NoMatchingPairError, redact

    out = Path(args.out)
    products = [p for p in cems.list_products(cems.fetch_activation(args.activation)) if p.aoi == args.aoi]
    if not products:
        raise LookupError(f"{args.activation} AOI {args.aoi} has no delineation product")
    refs = []
    for product in products:
        d = cems.download_product(product, out / "cems")
        flood, region = cems.observed_flood(d), scored_region(d)
        label = _write_geojson(out / "labels" / f"{product.tag}_flood.geojson", list(flood.geometry))
        aoi = _write_geojson(out / "labels" / f"{product.tag}_region.geojson", [region])
        refs.append({"tag": product.tag, "dir": str(d), "label": str(label), "region": str(aoi),
                     "time": max(product.acquisition_times).isoformat(), "sensor": product.images[0][0],
                     "flood_km2_in_region": float(flood.to_crs(32632).area.sum() / 1e6)})
    union = cems.aoi_geometry(Path(refs[0]["dir"]))
    sources = IngestSources.open()
    passes = find_passes(sources.client, union, args.start, args.end)
    cfg = load_config(args.config, overlays=args.overlay)
    tiles = [(t, b) for t, b in tiles_for(union.bounds) if _intersects(union, b)]
    report: dict[str, Any] = {"activation": args.activation, "aoi": args.aoi, "products": refs, "tiles": tiles,
                              "reference_before": str(args.reference_before), "passes": []}
    for p in passes:
        nearest = min(refs, key=lambda r: abs((datetime.fromisoformat(r["time"]) - p.when).total_seconds()))
        entry = {"tag": p.tag, "when": p.when.isoformat(), "relative_orbit": p.relative_orbit,
                 "orbit_state": p.orbit_state, "label_product": nearest["tag"], "tiles": {}}
        for tile_id, bbox in tiles:
            event = EventConfig(event_id=f"{p.tag}_{tile_id}", bbox=tuple(bbox), flood_start=p.when.date(),
                                flood_end=p.when.date(), label_path=Path(nearest["label"]),
                                aoi_path=Path(nearest["region"]), relative_orbit=p.relative_orbit,
                                orbit_state=p.orbit_state, reference_before=args.reference_before)
            done = out / "passes" / p.tag / "events" / event.event_id / "event.json"
            try:
                if not done.is_file():
                    ingest_event(event, out / "passes" / p.tag, cfg, sources=sources)
                entry["tiles"][tile_id] = "ok"
            except NoMatchingPairError as exc:
                entry["tiles"][tile_id] = f"dropped: {exc}"
            except Exception as exc:  # noqa: BLE001 - one tile must not abort the case
                entry["tiles"][tile_id] = f"failed: {type(exc).__name__}: {redact(str(exc))}"
            LOGGER.info("%s %s: %s", p.tag, tile_id, entry["tiles"][tile_id][:120])
        report["passes"].append(entry)
    (out / "case.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _intersects(geometry: Any, bbox: Sequence[float]) -> bool:
    from shapely.geometry import box

    return bool(geometry.intersects(box(*bbox)))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@dataclass
class _Model:
    name: str
    predict: Any          # (stack_path) -> probability (H, W) float, NaN = none
    tau: float


def _learned(run_dir: Path, device: Any) -> _Model:
    from floodfusion.evaluate import _load
    from floodfusion.predict import _resolve_run, predict_stack

    weights, evaluation = _resolve_run(run_dir)
    sar_only = bool(evaluation.get("force_sar_only"))
    cal = evaluation["calibration"]
    cfg, _, model, _, _ = _load(weights, None, device, sar_only)

    def run(stack: Path) -> np.ndarray:
        prob, _ = predict_stack(model, cfg, stack, temperature=cal["temperature"], offset=cal["offset"],
                                device=device, force_sar_only=sar_only)
        return prob

    return _Model("fusion, SAR only" if sar_only else "fusion (cross-attention)", run, cfg.eval.decision_threshold)


def _baseline(run_dir: Path, configs_dir: Path, overlays: Sequence[Path]) -> _Model:
    from floodfusion.baseline_runner import flood_map, load_scene, threshold_surface
    from floodfusion.config import load_config

    evaluation = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    cfg = load_config(configs_dir / "row02_baseline_adaptive.yaml", overlays=overlays)
    offset = float(evaluation["operating_offset_db"])

    def run(stack: Path) -> np.ndarray:
        scene = load_scene(stack, cfg)
        surface, _ = threshold_surface(scene, cfg)
        return flood_map(scene, surface, cfg, offset_db=offset).astype(np.float32)

    return _Model("physics baseline (row 2)", run, 0.5)


def _score(pred: np.ndarray, label: np.ndarray, scored: np.ndarray) -> ConfusionMatrix:
    p, y = pred[scored], label[scored]
    return ConfusionMatrix(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum()))


def _tile_layers(case_dir: Path, case: Mapping[str, Any], tile_id: str, models: Sequence[_Model]) -> dict[str, Any] | None:
    """Every pass's valid mask and predictions on one tile, and every product rasterised once."""
    from floodfusion.config import ValidityFlag
    from floodfusion.data.chipping import read_event_stack
    from floodfusion.data.encoding import ChipBand
    from floodfusion.data.ingest import rasterise_vector

    grid, passes = None, {}
    for p in case["passes"]:
        if p["tiles"].get(tile_id) != "ok":
            continue
        stack_path = case_dir / "passes" / p["tag"] / "events" / f"{p['tag']}_{tile_id}" / "stack.tif"
        raw, g, _ = read_event_stack(stack_path)
        if grid is not None and (g.transform, g.width, g.height) != (grid.transform, grid.width, grid.height):
            raise ValueError(f"{tile_id}: passes were ingested on different grids")
        grid = g
        meta = json.loads(stack_path.with_name("event.json").read_text(encoding="utf-8"))
        # Valid SAR: no shadow, layover, no data or permanent water (the AOI bit is the product's business).
        valid = (raw[ChipBand.VALIDITY].astype(np.int64) & ~int(ValidityFlag.OUTSIDE_AOI)) == 0
        passes[p["tag"]] = {"valid": valid, "optical": bool(meta.get("has_optical")),
                            "pred": {m.name: np.nan_to_num(m.predict(stack_path), nan=0.0) > m.tau for m in models}}
    if grid is None:
        return None
    refs = {r["tag"]: (rasterise_vector(Path(r["region"]), grid), rasterise_vector(Path(r["label"]), grid))
            for r in case["products"]}
    common = np.logical_and.reduce([v["valid"] for v in passes.values()] + [region for region, _ in refs.values()])
    return {"grid": grid, "passes": passes, "refs": refs, "common": common}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from floodfusion.train import resolve_device

    case_dir = Path(args.case)
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    models = [_learned(args.fusion_run, device)]
    if args.sar_only_run:
        models.append(_learned(args.sar_only_run, device))
    if args.baseline_run:
        models.append(_baseline(args.baseline_run, args.configs_dir, args.overlay))
    cms: dict[tuple[str, str, str], ConfusionMatrix] = defaultdict(lambda: ConfusionMatrix(0, 0, 0, 0))
    area: dict[str, float] = defaultdict(float)
    optical: dict[str, list[bool]] = defaultdict(list)
    for tile_id, _ in case["tiles"]:
        layers = _tile_layers(case_dir, case, tile_id, models)
        if layers is None:
            continue
        px_km2 = layers["grid"].pixel_size ** 2 / 1e6
        common = layers["common"]
        area["common"] += float(common.sum()) * px_km2
        for ref_tag, (_, label) in layers["refs"].items():
            area[f"cems:{ref_tag}"] += float((label & common).sum()) * px_km2
        for pass_tag, layer in layers["passes"].items():
            optical[pass_tag].append(layer["optical"])
            for name, pred in layer["pred"].items():
                area[f"{name}:{pass_tag}"] += float((pred & common).sum()) * px_km2
                for ref_tag, (region, label) in layers["refs"].items():
                    key = (pass_tag, ref_tag, name)
                    cms[key] = cms[key] + _score(pred, label, layer["valid"] & region)
    return _report(case, models, cms, area, optical, Path(args.out))


def _report(case: Mapping[str, Any], models: Sequence[_Model], cms: Mapping[tuple[str, str, str], ConfusionMatrix],
            area: Mapping[str, float], optical: Mapping[str, Sequence[bool]], out: Path) -> dict[str, Any]:
    from floodfusion.runs import write_json

    products = case["products"]
    lines = [
        f"# {case['activation']} AOI {case['aoi']} — every Sentinel-1 pass against CEMS's own delineations",
        "",
        "Zero-shot: the models were trained on Sen1Floods11 only. Inputs from Planetary Computer; pre-event "
        f"reference forced before {case['reference_before']} (the region had flooded on 2-3 May as well). Each "
        "pass is scored against **every** CEMS product on the area that product observed (its AOI inside its "
        "post-event image footprints, minus 'not analysed'), on valid SAR pixels outside permanent water. "
        "CEMS products: © European Union, Copernicus Emergency Management Service.",
        "",
        "## CEMS products", "",
        "| Product | Sensor | Acquired (UTC) | Flood in its region |", "| --- | --- | --- | --- |",
        *[f"| {r['tag']} | {r['sensor']} | {r['time'][:16]} | {r['flood_km2_in_region']:.1f} km² |" for r in products],
        "",
        "## IoU, pass × product (rows: Sentinel-1 passes; the nearest product in time is **bold**)", "",
    ]
    result: dict[str, Any] = {"iou": {}, "area_km2": {}, "time_offset_h": {}}
    for m in models:
        lines += [f"**{m.name}**", "", "| Pass | Optical | " + " | ".join(r["tag"].split("_", 2)[-1] for r in products) + " |",
                  "| --- | --- |" + " --- |" * len(products)]
        for p in case["passes"]:
            when = datetime.fromisoformat(p["when"])
            cells = []
            for r in products:
                cm = cms.get((p["tag"], r["tag"], m.name))
                iou = cm.iou if cm is not None and (cm.tp + cm.fp + cm.fn) else math.nan
                hours = (when - datetime.fromisoformat(r["time"])).total_seconds() / 3600
                result["iou"].setdefault(m.name, {}).setdefault(p["tag"], {})[r["tag"]] = iou
                result["time_offset_h"].setdefault(p["tag"], {})[r["tag"]] = hours
                text = f"{iou:.3f} ({hours:+.0f} h)" if math.isfinite(iou) else "—"
                cells.append(f"**{text}**" if r["tag"] == p["label_product"] else text)
            opt = optical.get(p["tag"], [])
            lines.append(f"| {p['tag']} | {sum(opt)}/{len(opt)} tiles | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Flooded area on the region every pass and every product observed (recession)", "",
              "| Map | Time (UTC) | Flooded km² |", "| --- | --- | --- |"]
    timeline = [(r["time"], f"CEMS {r['tag'].split('_', 2)[-1]} ({r['sensor']})", area.get(f"cems:{r['tag']}", math.nan))
                for r in products]
    timeline += [(p["when"], f"{m.name}, {p['tag']}", area.get(f"{m.name}:{p['tag']}", math.nan))
                 for p in case["passes"] for m in models]
    for when, label, km2 in sorted(timeline):
        lines.append(f"| {label} | {when[:16]} | {km2:.1f} |")
    result["area_km2"] = {label: km2 for _, label, km2 in timeline}
    lines += ["", f"Common region: {area.get('common', math.nan):.0f} km² (valid SAR on every pass, observed by every "
              "CEMS product).", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    write_json(out.with_suffix(".json"), {**result, "case": {k: case[k] for k in ("activation", "aoi", "reference_before")}})
    LOGGER.info("wrote %s", out)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A CEMS activation, every Sentinel-1 pass, against CEMS's maps.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--activation", default="EMSR664")
    prep.add_argument("--aoi", type=int, default=1)
    prep.add_argument("--start", type=date.fromisoformat, default=date(2023, 5, 16))
    prep.add_argument("--end", type=date.fromisoformat, default=date(2023, 5, 23))
    prep.add_argument("--reference-before", type=date.fromisoformat, default=date(2023, 5, 1))
    prep.add_argument("--out", type=Path, default=Path("data/emsr664"))
    prep.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    prep.add_argument("--overlay", type=Path, action="append", default=[])
    go = sub.add_parser("run")
    go.add_argument("--case", type=Path, default=Path("data/emsr664"))
    go.add_argument("--fusion-run", type=Path, required=True)
    go.add_argument("--sar-only-run", type=Path, default=None)
    go.add_argument("--baseline-run", type=Path, default=None)
    go.add_argument("--configs-dir", type=Path, default=Path("configs"))
    go.add_argument("--overlay", type=Path, action="append", default=[])
    go.add_argument("--device", default=None)
    go.add_argument("--out", type=Path, default=Path("reports/emsr664_forli.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare(args) if args.command == "prepare" else run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
