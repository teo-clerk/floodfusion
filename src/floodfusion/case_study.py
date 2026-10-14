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
observed: its AOI inside the footprints of the images **its polygons came from** (a
monitoring product delineates its newest image only), minus "not analysed".

Two scorings are reported. The **common region** — valid SAR on every pass and observed by
every product — makes the pass x product matrix and the flooded-area series comparable
across rows and columns; each pass's **own coverage** (larger) is reported beside it with
its scored area and flood prevalence. The pass x product matrix is the point: agreement
should peak at the product nearest in time and fall as the flood recedes. A Sentinel-1 map
compared with a delineation taken 12 hours earlier is a comparison of two instants; a 3 m
X-band delineation and a 10 m C-band map do not share a minimum mapping unit; and τ* with
its prior correction comes from Sen1Floods11, so the oracle-threshold IoU is shown too.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import ConfusionMatrix

__all__ = ["Pass", "find_passes", "full_coverage_passes", "main", "scored_region", "tiles_for"]

LOGGER = logging.getLogger(__name__)

#: ~16 x 17 km tiles at 44 N: ~2.7 Mpx each, ~0.6 GB peak during ingest.
TILE_DEG = (0.2, 0.15)
_ACQUISITION_GAP_S = 180
#: Thresholds for the oracle-threshold IoU of the learned models.
_GRID = tuple(round(i / 20, 2) for i in range(1, 20))


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

    from floodfusion.data.stac import datetime_of

    result = client.search(collections=["sentinel-1-rtc"], bbox=list(geometry.bounds),
                           datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z", max_items=200)
    items = [i if isinstance(i, dict) else i.to_dict() for i in (result.items() if hasattr(result, "items") else result)]
    items = [i for i in items if i.get("geometry") is None or shape(i["geometry"]).intersects(geometry)]
    passes: list[Pass] = []
    for item in sorted(items, key=datetime_of):
        when = datetime_of(item)
        p = item["properties"]
        candidate = Pass(when, int(p["sat:relative_orbit"]), str(p["sat:orbit_state"]).lower())
        if passes and passes[-1].relative_orbit == candidate.relative_orbit and \
                (when - passes[-1].when).total_seconds() <= _ACQUISITION_GAP_S:
            continue  # the next frame of the same acquisition
        passes.append(candidate)
    return passes


def tiles_for(bounds: Sequence[float], tile_deg: Sequence[float] = TILE_DEG) -> list[tuple[str, tuple[float, ...]]]:
    """Regular lon/lat tiles covering ``bounds``; ids ``t<row>_<col>`` from the north-west."""
    west, south, east, north = bounds
    # (12.1 - 11.7) / 0.2 is 2.0000000000000018 in binary: without the tolerance a sliver
    # column of empty tiles appears.
    nx = math.ceil((east - west) / tile_deg[0] - 1e-9)
    ny = math.ceil((north - south) / tile_deg[1] - 1e-9)
    out = []
    for r in range(ny):
        for c in range(nx):
            x0, y1 = west + c * tile_deg[0], north - r * tile_deg[1]
            out.append((f"t{r:02d}_{c:02d}", (x0, max(south, y1 - tile_deg[1]), min(east, x0 + tile_deg[0]), y1)))
    return out


def scored_region(product_dir: Path) -> Any:
    """Where a product observed: its AOI inside the footprints of the images its polygons came from.

    A monitoring product's footprint layer also lists earlier images; only those that produced
    its flood polygons (``dmg_src_id``) count — elsewhere the product says nothing. A product
    with no flood polygon falls back to its post-event images. Raises when no footprint matches.
    """
    import geopandas as gpd

    from floodfusion.data.cems import aoi_geometry, layer_path, observed_flood, source_times

    aoi = aoi_geometry(product_dir)
    path = layer_path(product_dir, "imageFootprintA")
    if path is None:
        raise FileNotFoundError(f"{product_dir}: no imageFootprintA layer — cannot tell where the product observed")
    frame = gpd.read_file(path)
    frame = frame.set_crs(4326) if frame.crs is None else frame.to_crs(4326)
    flood = observed_flood(product_dir)
    used = {int(s) for s in flood.get("dmg_src_id", []) if s is not None and float(s) == float(s)}
    sources = used or set(source_times(product_dir))
    kind = frame.get("obj_type", gpd.pd.Series([""] * len(frame))).astype(str)
    src = frame.get("or_src_id", gpd.pd.Series([None] * len(frame))).astype(float)
    seen = frame[(kind == "Image Footprint") & src.isin([float(s) for s in sources])]
    if seen.empty:
        raise ValueError(f"{product_dir.name}: no image footprint for source images {sorted(sources)}")
    blind = frame[kind.str.contains("Not Analysed", case=False) & src.isin([float(s) for s in sources])]
    region = aoi.intersection(seen.geometry.union_all())
    return region.difference(blind.geometry.union_all()) if len(blind) else region


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def _write_geojson(path: Path, geometries: Sequence[Any]) -> Path:
    import geopandas as gpd

    frame = gpd.GeoDataFrame([{}] * len(geometries), geometry=list(geometries), crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    return path


def _reference(product: Any, product_dir: Path, out: Path, area_epsg: int) -> dict[str, Any] | None:
    from floodfusion.data import cems

    flood = cems.observed_flood(product_dir)
    times = [t for t in flood.get("source_time", []) if t is not None] or list(product.acquisition_times)
    if not times:
        LOGGER.warning("%s: no acquisition time in the record or the source table — skipped", product.tag)
        return None
    region = scored_region(product_dir)
    label = _write_geojson(out / "labels" / f"{product.tag}_flood.geojson", list(flood.geometry))
    aoi = _write_geojson(out / "labels" / f"{product.tag}_region.geojson", [region])
    sensor = product.images[0][0] if product.images else "unknown"
    return {"tag": product.tag, "dir": str(product_dir), "label": str(label), "region": str(aoi),
            "time": max(times).isoformat(), "sensor": sensor,
            "flood_km2": float(flood.to_crs(area_epsg).area.sum() / 1e6),
            "region_km2": float(gpd_area_km2(region, area_epsg))}


def gpd_area_km2(geometry: Any, epsg: int) -> float:
    import geopandas as gpd

    return float(gpd.GeoSeries([geometry], crs=4326).to_crs(epsg).area.sum() / 1e6)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from floodfusion.config import load_config
    from floodfusion.data import cems
    from floodfusion.data.chipping import utm_epsg_for
    from floodfusion.data.ingest import EventConfig, IngestSources, ingest_event
    from floodfusion.data.stac import NoMatchingPairError, redact

    out = Path(args.out)
    products = [p for p in cems.list_products(cems.fetch_activation(args.activation)) if p.aoi == args.aoi]
    if not products:
        raise LookupError(f"{args.activation} AOI {args.aoi} has no delineation product")
    dirs = [cems.download_product(p, out / "cems") for p in products]
    union = cems.aoi_geometry(dirs[0])
    area_epsg = utm_epsg_for(union.centroid.x, union.centroid.y)
    refs = [r for p, d in zip(products, dirs) if (r := _reference(p, d, out, area_epsg)) is not None]
    sources = IngestSources.open()
    passes = find_passes(sources.client, union, args.start, args.end)
    cfg = load_config(args.config, overlays=args.overlay)
    tiles = [(t, b) for t, b in tiles_for(union.bounds) if _intersects(union, b)]
    report: dict[str, Any] = {"activation": args.activation, "aoi": args.aoi, "products": refs, "tiles": tiles,
                              "reference_before": str(args.reference_before), "area_epsg": area_epsg, "passes": []}
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
    predict: Callable[[Path], np.ndarray]   # stack -> probability (H, W), NaN = none
    tau: float
    calibrated: bool = True                 # a probability (oracle threshold meaningful) or a 0/1 map


#: The SAR-only evaluation of a model with an optical input, beside its run: row 9's is row 11.
SAR_ONLY_SIBLINGS = {"row09_fusion_crossattn": "row11_fusion_eval_sar_only",
                     "cmp_terramind_small": "cmp_terramind_small__eval_sar_only",
                     **{f"cmp_{a}": f"cmp_{a}__eval_sar_only" for a in ("terramind_base_fair",
                                                                         "terramind_base_fair_frozen",
                                                                         "terramind_small_fair")},
                     **{n: f"{n}__eval_sar_only" for n in ("row20_fusion_context", "row21_fusion_context_seasonal",
                                                           "row22_fusion_context_aef")}}


def sar_only_calibration(run_dir: Path) -> dict[str, float] | None:
    """The calibration a model was given for its no-optical mode (fitted on SAR-only validation logits).

    A model with an optical input is two predictors: with optical and without. v1.0 calibrates
    both (row 9 and row 11); a window without usable optical must be read with the second.
    """
    sibling = SAR_ONLY_SIBLINGS.get(run_dir.parent.name)
    if sibling is None:
        return None                                  # a model without an optical input: one predictor
    path = run_dir.parent.parent / sibling / run_dir.name / "eval_test.json"
    if not path.is_file():
        # Silently reading its no-optical windows with the with-optical calibration would change scores unseen.
        raise FileNotFoundError(f"{run_dir} needs its SAR-only calibration {path}: evaluate the sibling first")
    return json.loads(path.read_text(encoding="utf-8"))["calibration"]


def recalibrate_without_optical(prob: np.ndarray, optical_used: np.ndarray, cal: Mapping[str, float],
                                sar_cal: Mapping[str, float]) -> np.ndarray:
    """Re-read the windows that ran without optical with the SAR-only calibration.

    ``prob = sigmoid((z + c) / T)`` with the with-optical (T, c); the raw score is recovered as
    ``z = T logit(prob) - c`` and re-calibrated as ``sigmoid((z + c') / T')``. One forward pass.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip(prob.astype(np.float64), 1e-12, 1 - 1e-12)
        z = cal["temperature"] * np.log(p / (1 - p)) - cal["offset"]
        sar = 1.0 / (1.0 + np.exp(-(z + sar_cal["offset"]) / sar_cal["temperature"]))
    return np.where(optical_used | ~np.isfinite(prob), prob, sar).astype(np.float32)


def _learned(run_dir: Path, device: Any) -> _Model:
    from floodfusion.evaluate import load_run
    from floodfusion.predict import _resolve_run, predict_stack

    weights, evaluation = _resolve_run(run_dir)
    sar_only = bool(evaluation.get("force_sar_only"))
    cal = evaluation["calibration"]
    sar_cal = None if sar_only else sar_only_calibration(run_dir)
    cfg, _, model, _, _ = load_run(weights, None, device, sar_only)

    def run(stack: Path) -> np.ndarray:
        prob, optical_used = predict_stack(model, cfg, stack, temperature=cal["temperature"], offset=cal["offset"],
                                           device=device, force_sar_only=sar_only)
        return prob if sar_cal is None else recalibrate_without_optical(prob, optical_used, cal, sar_cal)

    return _Model("fusion, SAR only" if sar_only else "fusion (cross-attention)", run, cfg.eval.decision_threshold)


def _baseline(run_dir: Path, configs_dir: Path, overlays: Sequence[Path]) -> _Model:
    from floodfusion.baseline_runner import flood_map, load_scene, threshold_surface
    from floodfusion.config import load_config

    evaluation = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    record = run_dir / "run.json"
    if record.is_file():   # the run's own resolved configuration (row 2 or a v1.1 variant of it)
        from floodfusion.config import config_from_dict

        cfg = config_from_dict(json.loads(record.read_text(encoding="utf-8"))["config"])
    else:
        cfg = load_config(configs_dir / "row02_baseline_adaptive.yaml", overlays=overlays)
    offset = float(evaluation["operating_offset_db"])

    def run(stack: Path) -> np.ndarray:
        # A case tile is a ~16 x 17 km scene: its own histogram is the scene context, so the event-pooled
        # fallback (threshold_fallback: event) does not apply here; a tile without a bimodal sub-tile
        # falls back to its own Otsu. Deployment has no label: the threshold comes from every valid SAR pixel. (The benchmark
        # runner scores labelled pixels; a case tile's label is a CEMS delineation, often absent.)
        scene = load_scene(stack, cfg)
        valid = np.isfinite(scene.vv_post_db) & ~scene.layover & ~scene.shadow & ~scene.permanent
        if int(valid.sum()) < 2:
            LOGGER.warning("%s: no usable SAR pixels; the physics baseline maps no flood", stack.parent.name)
            return np.zeros(scene.vv_post_db.shape, np.float32)
        surface, _ = threshold_surface(replace(scene, scored=valid), cfg)
        return flood_map(scene, surface, cfg, offset_db=offset).astype(np.float32)

    return _Model("physics baseline (row 2)", run, 0.5, calibrated=False)


def _score(pred: np.ndarray, label: np.ndarray, scored: np.ndarray) -> ConfusionMatrix:
    p, y = pred[scored], label[scored]
    return ConfusionMatrix(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum()))


def _scores(prob: np.ndarray, model: _Model, label: np.ndarray, scored: np.ndarray) -> list[ConfusionMatrix]:
    """Confusion at τ* first, then (calibrated models only) on the oracle grid."""
    thresholds = (model.tau, *(_GRID if model.calibrated else ()))
    return [_score(prob > t, label, scored) for t in thresholds]


def full_coverage_passes(case: Mapping[str, Any]) -> set[str]:
    """The passes covering the most tiles: the common region is defined over these.

    A pass that saw only the edge of the AOI (EMSR664: 18 May, 4 of 16 tiles) would otherwise
    shrink the region every pass observed to nothing; it is scored on its own coverage instead.
    """
    counts = {p["tag"]: sum(v == "ok" for v in p["tiles"].values()) for p in case["passes"]}
    best = max(counts.values(), default=0)
    return {tag for tag, n in counts.items() if n == best}


def _tile_layers(case_dir: Path, case: Mapping[str, Any], tile_id: str, models: Sequence[_Model],
                 full: set[str]) -> dict[str, Any] | None:
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
                            "prob": {m.name: np.nan_to_num(m.predict(stack_path), nan=0.0) for m in models}}
    if grid is None:
        return None
    refs = {r["tag"]: (rasterise_vector(Path(r["region"]), grid), rasterise_vector(Path(r["label"]), grid))
            for r in case["products"]}
    common = (np.logical_and.reduce([passes[tag]["valid"] for tag in sorted(full)]) if full and full <= set(passes)
              else np.zeros((grid.height, grid.width), dtype=bool))
    return {"grid": grid, "passes": passes, "refs": refs, "common": common}


@dataclass
class _Tally:
    common: dict[tuple[str, str, str], list[ConfusionMatrix]]
    own: dict[tuple[str, str, str], list[ConfusionMatrix]]
    area: dict[str, float]
    optical: dict[str, list[bool]]

    def add(self, store: dict[tuple[str, str, str], list[ConfusionMatrix]], key: tuple[str, str, str],
            cms: list[ConfusionMatrix]) -> None:
        store[key] = cms if key not in store else [a + b for a, b in zip(store[key], cms)]


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
    tally = _Tally({}, {}, defaultdict(float), defaultdict(list))
    full = full_coverage_passes(case)
    nearest = {p["tag"]: p["label_product"] for p in case["passes"]}
    hero: dict[str, tuple[float, str]] = {}
    for tile_id, _ in case["tiles"]:
        layers = _tile_layers(case_dir, case, tile_id, models, full)
        if layers is None:
            continue
        px_km2 = layers["grid"].pixel_size ** 2 / 1e6
        everywhere = layers["common"] & np.logical_and.reduce([region for region, _ in layers["refs"].values()])
        tally.area["common"] += float(everywhere.sum()) * px_km2
        for ref_tag, (_, label) in layers["refs"].items():
            tally.area[f"cems:{ref_tag}"] += float((label & everywhere).sum()) * px_km2
        for pass_tag, layer in layers["passes"].items():
            tally.optical[pass_tag].append(layer["optical"])
            on_common = pass_tag in full
            for m in models:
                prob = layer["prob"][m.name]
                if on_common:
                    tally.area[f"{m.name}:{pass_tag}"] += float(((prob > m.tau) & everywhere).sum()) * px_km2
                for ref_tag, (region, label) in layers["refs"].items():
                    if on_common:
                        tally.add(tally.common, (pass_tag, ref_tag, m.name),
                                  _scores(prob, m, label, layers["common"] & region))
                    tally.add(tally.own, (pass_tag, ref_tag, m.name), _scores(prob, m, label, layer["valid"] & region))
            if on_common:  # hero candidates: the tile with the most CEMS flood seen by this pass
                region, label = layers["refs"][nearest[pass_tag]]
                flooded = float((label & region & layer["valid"]).sum())
                hero[pass_tag] = max(hero.get(pass_tag, (-1.0, "")), (flooded, tile_id))
    result = _report(case, models, tally, Path(args.out))
    if hero and args.assets is not None:
        # The full pass with optical on the most tiles: the case the fusion model exists for.
        hero_pass = max(hero, key=lambda tag: (sum(tally.optical[tag]), hero[tag][0]))
        _figures(case, case_dir, hero_pass, hero[hero_pass][1], models[0], tally, Path(args.assets))
    return result


# ---------------------------------------------------------------------------
# Report and figures
# ---------------------------------------------------------------------------
def _md(text: str) -> str:
    """Strings from the API in a Markdown table: no pipes, no line breaks."""
    return str(text).replace("|", "/").replace("\n", " ").replace("\r", " ")


def _iou_cell(cms: list[ConfusionMatrix] | None, calibrated: bool) -> tuple[str, float, float]:
    if not cms or not (cms[0].tp + cms[0].fp + cms[0].fn):
        return "—", math.nan, math.nan
    tau = cms[0].iou
    oracle = max((c.iou for c in cms[1:] if not math.isnan(c.iou)), default=math.nan) if calibrated else math.nan
    text = f"{tau:.3f}" + (f" / {oracle:.3f}" if calibrated and math.isfinite(oracle) else "")
    return text, tau, oracle


def _report(case: Mapping[str, Any], models: Sequence[_Model], tally: _Tally, out: Path) -> dict[str, Any]:
    from floodfusion.runs import write_json

    products = case["products"]
    short = {r["tag"]: r["tag"].split("_", 2)[-1] for r in products}
    lines = [
        f"# {case['activation']} AOI {case['aoi']} — every Sentinel-1 pass against CEMS's own delineations",
        "",
        "Zero-shot: the models were trained on Sen1Floods11 only; τ* and the prior correction come from its "
        "validation set and need not transfer to this scene, so each learned cell shows **IoU at τ* / IoU at the "
        "oracle threshold** (the latter is not deployable; the gap between them is the threshold transfer). Inputs "
        f"from Planetary Computer; the pre-event reference is forced before {case['reference_before']} (the region "
        "had flooded on 2-3 May as well). A product's region is its AOI inside the footprints of the images its "
        "polygons came from, minus 'not analysed'. Scored on valid SAR pixels outside permanent water. CEMS "
        "products: © European Union, Copernicus Emergency Management Service.",
        "",
        "## CEMS products", "",
        "| Product | Sensor | Acquired (UTC) | Region observed | Flood in it |", "| --- | --- | --- | --- | --- |",
        *[f"| {short[r['tag']]} | {_md(r['sensor'])} | {r['time'][:16]} | {r['region_km2']:.0f} km² | {r['flood_km2']:.1f} km² |"
          for r in products],
        "",
    ]
    result: dict[str, Any] = {"common": {}, "own": {}, "area_km2": {}, "time_offset_h": {}}
    full = full_coverage_passes(case)
    partial = [p["tag"] for p in case["passes"] if p["tag"] not in full]
    for scope, store, title in (("common", tally.common, "on the region common to every full-coverage pass and "
                                                         "every product"),
                                ("own", tally.own, "on each pass's own coverage")):
        lines += [f"## IoU, pass × product, {title} (nearest product in time in **bold**)", ""]
        for m in models:
            head = "| Pass | Optical tiles | " + " | ".join(short[r["tag"]] for r in products) + " |"
            lines += [f"**{m.name}**" + ("" if m.calibrated else " (a 0/1 rule: IoU at its operating offset)"), "",
                      head, "| --- | --- |" + " --- |" * len(products)]
            for p in case["passes"]:
                when = datetime.fromisoformat(p["when"])
                cells = []
                for r in products:
                    text, tau, oracle = _iou_cell(store.get((p["tag"], r["tag"], m.name)), m.calibrated)
                    hours = (when - datetime.fromisoformat(r["time"])).total_seconds() / 3600
                    result[scope].setdefault(m.name, {}).setdefault(p["tag"], {})[r["tag"]] = {"iou_tau": tau,
                                                                                              "iou_oracle": oracle}
                    result["time_offset_h"].setdefault(p["tag"], {})[r["tag"]] = hours
                    cell = f"{text} ({hours:+.0f} h)" if text != "—" else "—"
                    cells.append(f"**{cell}**" if r["tag"] == p["label_product"] else cell)
                opt = tally.optical.get(p["tag"], [])
                lines.append(f"| {p['tag']} | {sum(opt)}/{len(opt)} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## Scored area and flood prevalence per pass (own coverage, nearest product)", "",
              "| Pass | Nearest product | Scored km² | CEMS flood prevalence |", "| --- | --- | --- | --- |"]
    for p in case["passes"]:
        cms = tally.own.get((p["tag"], p["label_product"], models[0].name))
        if cms:
            c = cms[0]
            n = c.tp + c.fp + c.fn + c.tn
            lines.append(f"| {p['tag']} | {short[p['label_product']]} | {n / 1e4:.0f} | {(c.tp + c.fn) / max(n, 1):.3f} |")
    lines += ["", "## Flooded area on the region every full-coverage pass and every product observed (recession)", "",
              "| Map | Time (UTC) | Flooded km² |", "| --- | --- | --- |"]
    timeline = [(r["time"], f"CEMS {short[r['tag']]} ({_md(r['sensor'])})", tally.area.get(f"cems:{r['tag']}", math.nan))
                for r in products]
    timeline += [(p["when"], f"{m.name}, {p['tag']}", tally.area.get(f"{m.name}:{p['tag']}", math.nan))
                 for p in case["passes"] if p["tag"] in full for m in models]
    for when, label, km2 in sorted(timeline):
        lines.append(f"| {label} | {when[:16]} | {km2:.1f} |")
    result["area_km2"] = {label: km2 for _, label, km2 in timeline}
    lines += ["", f"Common region: {tally.area.get('common', math.nan):.0f} km² (valid SAR on every full-coverage pass, "
              "observed by every CEMS product)."
              + (f" Passes that covered part of the AOI are scored on their own coverage only: {', '.join(partial)}."
                 if partial else ""), "", "![EMSR664 Forlì](../assets/forli_hero.png)", "",
              "![recession](../assets/forli_recession.png)", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    write_json(out.with_suffix(".json"), {**result, "case": {k: case[k] for k in ("activation", "aoi", "reference_before")}})
    LOGGER.info("wrote %s", out)
    return result


def _figures(case: Mapping[str, Any], case_dir: Path, pass_tag: str, tile_id: str, model: _Model, tally: _Tally,
             assets: Path) -> None:
    """The hero map (``pass_tag`` against its nearest product) and the recession series."""
    from floodfusion.data.chipping import read_event_stack
    from floodfusion.data.encoding import ChipBand
    from floodfusion.data.ingest import rasterise_vector
    from floodfusion.figures import _pyplot

    plt = _pyplot()
    first = next(p for p in case["passes"] if p["tag"] == pass_tag)
    stack_path = case_dir / "passes" / first["tag"] / "events" / f"{first['tag']}_{tile_id}" / "stack.tif"
    raw, grid, _ = read_event_stack(stack_path)
    ref = next(r for r in case["products"] if r["tag"] == first["label_product"])
    region, label = rasterise_vector(Path(ref["region"]), grid), rasterise_vector(Path(ref["label"]), grid)
    vv = np.where(raw[ChipBand.VV_POST] == -32768, np.nan, raw[ChipBand.VV_POST] / 100.0)
    pred = np.nan_to_num(model.predict(stack_path), nan=0.0) > model.tau
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title(f"Sentinel-1 VV post (dB), {first['when'][:16]} UTC", fontsize=10)
    axes[1].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[1].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.7)
    axes[1].set_title(f"{model.name}: flood at τ*", fontsize=10)
    axes[2].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[2].imshow(np.ma.masked_where(~(label & region), label), cmap="winter", alpha=0.7)
    axes[2].set_title(f"CEMS {ref['tag'].split('_', 2)[-1]} ({_md(ref['sensor'])}, {ref['time'][11:16]} UTC)", fontsize=10)
    for ax in axes:
        ax.contour(region, levels=[0.5], colors="yellow", linewidths=0.6)
        ax.axis("off")
    fig.suptitle(f"{case['activation']} — zero-shot flood map vs the Copernicus EMS delineation (yellow: its region)",
                 fontsize=11)
    fig.tight_layout()
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / "forli_hero.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    cems_points = sorted((datetime.fromisoformat(r["time"]), tally.area.get(f"cems:{r['tag']}", math.nan))
                         for r in case["products"])
    ax.plot([t for t, _ in cems_points], [a for _, a in cems_points], "s-", color="tab:blue", label="CEMS delineations")
    full = full_coverage_passes(case)
    s1 = sorted((datetime.fromisoformat(p["when"]), tally.area.get(f"{model.name}:{p['tag']}", math.nan))
                for p in case["passes"] if p["tag"] in full)
    ax.plot([t for t, _ in s1], [a for _, a in s1], "o-", color="tab:red", label=f"{model.name} (Sentinel-1)")
    ax.set_ylabel(f"flooded km² ({tally.area.get('common', math.nan):.0f} km² seen by all)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(assets / "forli_recession.png", dpi=90)
    plt.close(fig)


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
    go.add_argument("--assets", type=Path, default=Path("assets"))
    go.add_argument("--out", type=Path, default=Path("reports/emsr664_forli.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare(args) if args.command == "prepare" else run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
