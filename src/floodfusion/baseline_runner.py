r"""Ablation rows 1--3: the non-learned physics baseline, scored like every learned row.

``P1-M2``. Run as::

    python -m floodfusion.baseline_runner --data-root data/synthetic --runs-root runs \
        --overlay configs/overlays/synthetic_demo.yaml

For each row config (``row01_otsu_global``, ``row02_baseline_adaptive``,
``row03_baseline_mndwi``) and each event stack holding validation or test chips:

1. Decode the stack; Refined-Lee filter VV pre and post **in linear power** (ENL 4.4,
   ``baseline.speckle_window``), back to dB.
2. Threshold surface: global Otsu (row 1) or the tile-adaptive surface with the explicit
   acceptance rule (rows 2--3); a scene where no tile is accepted falls back to global
   Otsu, and the fallback is recorded.
3. Sweep a **threshold offset** :math:`\delta \in [-10, 6]` dB: the PR curve of a hard
   classifier. The operating point is the offset minimising the expected cost
   :math:`r \cdot FN + FP` (``docs/00`` section 6, :math:`r = 3`) **on validation**, then
   applied unchanged to test — the same rule that places :math:`\tau^\star` for the networks.
4. Score per test basin over valid pixels of the evaluation-grid chips — the identical
   metric code path as the learned rows.
5. Row 2 only: the per-mask attribution table (``P1-M2.4``: slope, HAND, layover/shadow,
   permanent water, each disabled in turn) and AND versus OR (``docs/06``).

Outputs ``<runs_root>/<row name>/baseline/{run.json, eval_test.json}`` in the schema
:mod:`floodfusion.reporting` reads for every row.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ExperimentConfig, config_sha256, config_to_dict, load_config
from floodfusion.metrics import (
    ConfusionMatrix,
    average_precision,
    basin_bootstrap_ci,
    macro_iou,
    precision_recall_curve,
    sum_confusions,
)
from floodfusion.runs import write_json, write_run_record

__all__ = ["BASELINE_ROWS", "OFFSETS_DB", "SceneLayers", "main", "run_baseline_row"]

LOGGER = logging.getLogger(__name__)

BASELINE_ROWS = ("row01_otsu_global", "row02_baseline_adaptive", "row03_baseline_mndwi")
#: Threshold offsets of the PR sweep (dB). Asymmetric: a global Otsu threshold sits between the
#: land and water modes and its cost-optimal shift is downward, toward the water mode; the
#: range must contain the optimum, never clip it (an optimum on the edge is not an optimum).
OFFSETS_DB = tuple(float(v) for v in np.round(np.arange(-10.0, 6.0 + 1e-9, 0.5), 2))
_ENL = 4.4


@dataclass(frozen=True)
class SceneLayers:
    """One event, decoded and filtered, ready for the decision rule."""

    vv_post_db: np.ndarray
    vv_pre_db: np.ndarray
    mndwi: np.ndarray | None
    optical_valid: np.ndarray | None
    slope: np.ndarray
    hand: np.ndarray
    layover: np.ndarray
    shadow: np.ndarray
    permanent: np.ndarray
    label: np.ndarray
    scored: np.ndarray  # valid pixels with a label


def _filtered_db(db: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    from floodfusion.data.preprocessing import db_to_linear, linear_to_db, refined_lee

    if cfg.baseline.speckle_filter == "none":
        return db
    linear = np.nan_to_num(db_to_linear(np.where(np.isfinite(db), db, np.nan)), nan=0.0)
    filtered = refined_lee(linear, looks=_ENL, window=cfg.baseline.speckle_window)
    out = linear_to_db(filtered)
    return np.where(np.isfinite(db) & (out > -300), out, np.nan)


def load_scene(stack_path: Path, cfg: ExperimentConfig) -> SceneLayers:
    from floodfusion.config import ValidityFlag
    from floodfusion.data.chipping import CLEAR_QUALITY, read_event_stack
    from floodfusion.data.encoding import decode_physical

    raw, _, _ = read_event_stack(stack_path)
    chip = decode_physical(raw, cfg.grid, cfg.norm)
    v = chip.validity.astype(np.int64)
    has_optical = bool(np.isfinite(chip.mndwi).any())
    return SceneLayers(
        vv_post_db=_filtered_db(chip.gamma0_db[0], cfg), vv_pre_db=_filtered_db(chip.gamma0_db[2], cfg),
        mndwi=chip.mndwi if has_optical else None,
        optical_valid=(chip.optical_quality >= CLEAR_QUALITY) & np.isfinite(chip.mndwi) if has_optical else None,
        slope=np.nan_to_num(chip.slope_deg, nan=0.0), hand=np.nan_to_num(chip.hand_m, nan=0.0),
        layover=(v & int(ValidityFlag.LAYOVER)) > 0, shadow=(v & int(ValidityFlag.SHADOW)) > 0,
        permanent=(v & int(ValidityFlag.PERMANENT_WATER)) > 0,
        label=chip.label == 1.0, scored=(v == 0) & np.isfinite(chip.label),
    )


def threshold_surface(scene: SceneLayers, cfg: ExperimentConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Global Otsu (row 1) or the tile-adaptive surface, with the Otsu fallback recorded."""
    from floodfusion.baselines import otsu_threshold, tile_adaptive_threshold

    usable = scene.scored & np.isfinite(scene.vv_post_db)
    global_t = otsu_threshold(scene.vv_post_db[usable])
    if cfg.baseline.threshold_mode == "global_otsu":
        return np.full(scene.vv_post_db.shape, global_t), {"mode": "global_otsu", "otsu_db": global_t}
    b = cfg.baseline
    result = tile_adaptive_threshold(scene.vv_post_db, usable, tile_size=b.tile_size,
                                     min_valid_fraction=b.min_tile_valid_fraction, ashman_min=b.ashman_min,
                                     min_component_weight=b.min_component_weight,
                                     water_mode_max_db=b.water_mode_max_db)
    info = {"mode": "tile_adaptive", "accepted_fraction": result.accepted_fraction, "otsu_db": global_t}
    surface = result.threshold_surface
    if not np.isfinite(surface).any():
        LOGGER.warning("no tile accepted: falling back to the global Otsu threshold %.2f dB", global_t)
        return np.full(scene.vv_post_db.shape, global_t), {**info, "fallback": "global_otsu"}
    return np.where(np.isfinite(surface), surface, global_t), info


def flood_map(scene: SceneLayers, surface: np.ndarray, cfg: ExperimentConfig, *, offset_db: float = 0.0,
              disable: str | None = None, combine: str = "and") -> np.ndarray:
    """The ``docs/06`` 5.1 rule for this row, optionally with one exclusion disabled."""
    from floodfusion.baselines import baseline_flood_map

    b = cfg.baseline
    use_optical = b.use_optical_override and scene.mndwi is not None
    zeros = np.zeros_like(scene.layover)
    grow = None if cfg.baseline.threshold_mode == "global_otsu" else b.region_grow_margin_db
    return baseline_flood_map(
        np.nan_to_num(scene.vv_post_db, nan=0.0), np.nan_to_num(scene.vv_pre_db, nan=0.0), surface + offset_db,
        scene.mndwi if use_optical else None, scene.optical_valid if use_optical else None,
        scene.slope, scene.hand,
        zeros if disable == "layover_shadow" else scene.layover,
        zeros if disable == "layover_shadow" else scene.shadow,
        scene.permanent, delta_min_db=b.delta_min_db, tau_mndwi=b.tau_mndwi,
        slope_max_deg=math.inf if disable == "slope" else b.slope_max_deg,
        hand_max_m=math.inf if disable == "hand" else b.hand_max_m,
        use_change_term=b.use_change_term, combine=combine, grow_margin_db=grow,
        apply_terrain_masks=b.apply_terrain_masks,
        apply_permanent_water_mask=b.apply_permanent_water_mask and disable != "permanent_water",
    )


def score_windows(pred: np.ndarray, scene: SceneLayers, windows: Sequence[tuple[str, tuple[int, int, int, int]]],
                  ) -> dict[str, ConfusionMatrix]:
    """Per-basin confusion over the valid pixels of each chip window."""
    out: dict[str, ConfusionMatrix] = {}
    for basin, (col, row, w, h) in windows:
        sl = (slice(row, row + h), slice(col, col + w))
        m, p, y = scene.scored[sl], pred[sl], scene.label[sl]
        cm = ConfusionMatrix(tp=int((p & y & m).sum()), fp=int((p & ~y & m).sum()),
                             fn=int((~p & y & m).sum()), tn=int((~p & ~y & m).sum()))
        out[basin] = out[basin] + cm if basin in out else cm
    return out


def _merge(a: dict[str, ConfusionMatrix], b: Mapping[str, ConfusionMatrix]) -> dict[str, ConfusionMatrix]:
    return {k: a.get(k, ConfusionMatrix(0, 0, 0, 0)) + b.get(k, ConfusionMatrix(0, 0, 0, 0)) for k in set(a) | set(b)}


def _cost(per_basin: Mapping[str, ConfusionMatrix], r: float) -> float:
    pooled = sum_confusions(list(per_basin.values()))
    return r * pooled.fn + pooled.fp


def _windows_by_event(records: Sequence[Any]) -> dict[Path, list[tuple[str, tuple[int, int, int, int]]]]:
    out: dict[Path, list[tuple[str, tuple[int, int, int, int]]]] = defaultdict(list)
    for r in records:
        out[Path(r.raster_path)].append((r.basin_id, tuple(r.window)))
    return out


def run_baseline_row(cfg: ExperimentConfig, data_root: Path, out_dir: Path, *, attribution: bool = False,
                     seed: int = 0) -> dict[str, Any]:
    """Sweep, pick the cost-optimal offset on validation, score test; write the row outputs."""
    from floodfusion.train import load_training_data

    data = load_training_data(cfg, data_root)
    windows = {"val": _windows_by_event(data.val), "test": _windows_by_event(data.test)}
    sweep: dict[str, list[dict[str, ConfusionMatrix]]] = {s: [{} for _ in OFFSETS_DB] for s in windows}
    variants: dict[str, dict[str, ConfusionMatrix]] = defaultdict(dict)
    scene_info: dict[str, Any] = {}
    elapsed, n_scored_chips = 0.0, 0
    for path in sorted(set(windows["val"]) | set(windows["test"])):
        started = time.perf_counter()
        scene = load_scene(path, cfg)
        surface, info = threshold_surface(scene, cfg)
        scene_info[path.parent.name] = info
        for k, offset in enumerate(OFFSETS_DB):
            pred = flood_map(scene, surface, cfg, offset_db=float(offset))
            for split, by_event in windows.items():
                if path in by_event:
                    sweep[split][k] = _merge(sweep[split][k], score_windows(pred, scene, by_event[path]))
        elapsed += (time.perf_counter() - started) / len(OFFSETS_DB)
        n_scored_chips += len(windows["val"].get(path, [])) + len(windows["test"].get(path, []))
        if attribution:  # ~46 B/px per scene: keep the layers only when row 2's attribution needs them
            scene_info[path.parent.name]["scene"] = scene
            scene_info[path.parent.name]["surface"] = surface
    r = cfg.eval.cost_ratio
    best = int(np.argmin([_cost(pb, r) if pb else math.inf for pb in sweep["val"]]))
    offset = float(OFFSETS_DB[best])
    if best in (0, len(OFFSETS_DB) - 1):
        LOGGER.warning("%s: the cost-optimal offset %+.1f dB lies on the edge of the sweep", cfg.name, offset)
    if attribution and windows["test"]:
        variants = _attribution(windows["test"], scene_info, cfg, offset)
    for info in scene_info.values():
        info.pop("scene", None)
        info.pop("surface", None)
    result = _row_result(cfg, sweep, best, offset, seed, variants, scene_info, data)
    result["latency"] = {"device": "cpu", "latency_ms_median": 1e3 * elapsed / max(n_scored_chips, 1),
                         "note": "CPU wall time of one decision-rule evaluation per 256^2 chip, scene-amortised"}
    _write_outputs(cfg, data, out_dir, result)
    return result


def _attribution(test_windows: Mapping[Path, Any], scene_info: Mapping[str, Any], cfg: ExperimentConfig,
                 offset: float) -> dict[str, dict[str, ConfusionMatrix]]:
    """P1-M2.4: each exclusion disabled in turn, and the OR decision rule (docs/06 ex. 6)."""
    variants: dict[str, dict[str, ConfusionMatrix]] = defaultdict(dict)
    for path, wins in test_windows.items():
        scene, surface = scene_info[path.parent.name]["scene"], scene_info[path.parent.name]["surface"]
        no_change = replace(cfg, baseline=replace(cfg.baseline, use_change_term=False))
        for name, kwargs in (("all_masks", {}), ("no_slope", {"disable": "slope"}), ("no_hand", {"disable": "hand"}),
                             ("no_layover_shadow", {"disable": "layover_shadow"}),
                             ("no_permanent_water", {"disable": "permanent_water"}), ("or_rule", {"combine": "or"}),
                             ("no_change_term", {"_cfg": no_change})):
            rule_cfg = kwargs.pop("_cfg", cfg)
            pred = flood_map(scene, surface, rule_cfg, offset_db=offset, **kwargs)
            variants[name] = _merge(variants[name], score_windows(pred, scene, wins))
    return variants


def _row_result(cfg: ExperimentConfig, sweep: Mapping[str, Sequence[Mapping[str, ConfusionMatrix]]], best: int,
                offset: float, seed: int, variants: Mapping[str, Mapping[str, ConfusionMatrix]],
                scene_info: Mapping[str, Any], data: Any) -> dict[str, Any]:
    from floodfusion.data.records import chip_set_sha256
    from floodfusion.evaluate import _basin_summary, _macro
    from floodfusion.utils.env import git_state

    test = dict(sweep["test"][best])
    zero = list(OFFSETS_DB).index(0.0)
    pooled = [sum_confusions(list(pb.values())) for pb in sweep["test"]]
    try:
        ap = average_precision(*precision_recall_curve(tuple(pooled)))
        precisions, recalls = precision_recall_curve(tuple(pooled))
    except ValueError:
        ap, precisions, recalls = float("nan"), (), ()
    n_boot, alpha = cfg.eval.n_bootstrap, cfg.eval.bootstrap_alpha
    pooled_best = sum_confusions(list(test.values()))
    attribution = {name: {"macro_iou": macro_iou(dict(pb)), "macro_precision": _macro(pb, "precision"),
                          "macro_recall": _macro(pb, "recall")} for name, pb in variants.items()}
    return {
        "name": cfg.name, "ablation_row": cfg.ablation_row, "split": "test", "baseline": True,
        "operating_offset_db": offset, "operating_rule": f"argmin over validation of r*FN + FP, r = {cfg.eval.cost_ratio:g}",
        "macro_iou": macro_iou(test),
        "macro_iou_ci": list(basin_bootstrap_ci(test, n_boot=n_boot, alpha=alpha, seed=seed, statistic="macro")[1:]),
        "macro_f1": _macro(test, "f1"), "macro_precision": _macro(test, "precision"),
        "macro_recall": _macro(test, "recall"), "pooled_iou": pooled_best.iou,
        "pooled_iou_ci": list(basin_bootstrap_ci(test, n_boot=n_boot, alpha=alpha, seed=seed, statistic="pooled")[1:]),
        "pooled_f1": pooled_best.f1, "per_basin": _basin_summary(test),
        "macro_iou_frozen_offset0": macro_iou(dict(sweep["test"][zero])),
        "average_precision": ap, "pr_curve": {"precision": list(precisions), "recall": list(recalls)},
        "sweep_offsets_db": list(OFFSETS_DB),
        "sweep_val_macro_iou": [macro_iou(dict(pb)) if pb else float("nan") for pb in sweep["val"]],
        "sweep_test_macro_iou": [macro_iou(dict(pb)) if pb else float("nan") for pb in sweep["test"]],
        "mask_attribution": attribution, "scenes": dict(scene_info), "n_chips": len(data.test),
        "attribution_per_basin": {name: _basin_summary(dict(pb)) for name, pb in variants.items()},
        "manifest_sha256": data.manifest_sha256, "split_sha256": data.split_sha256,
        "test_chips_sha256": chip_set_sha256(data.test), "synthetic_data": data.is_synthetic,
        "evaluation_git": git_state(),
    }


def _write_outputs(cfg: ExperimentConfig, data: Any, out_dir: Path, result: Mapping[str, Any]) -> None:
    from floodfusion.utils.env import environment_record, git_state

    write_run_record(out_dir, {
        "name": cfg.name, "ablation_row": cfg.ablation_row, "architecture": "threshold_baseline",
        "config": config_to_dict(cfg), "config_sha256": config_sha256(config_to_dict(cfg)),
        "manifest_sha256": data.manifest_sha256, "synthetic_data": data.is_synthetic, "status": "finished",
        "trained": False, "environment": environment_record(), **git_state(),
    })
    write_json(out_dir / "eval_test.json", result)
    LOGGER.info("%s: test macro IoU %.3f [%.3f, %.3f] at offset %+.1f dB (frozen offset 0: %.3f)", cfg.name,
                result["macro_iou"], *result["macro_iou_ci"], result["operating_offset_db"],
                result["macro_iou_frozen_offset0"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablation rows 1-3: the physics baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--rows", nargs="*", default=list(BASELINE_ROWS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in args.rows:
        cfg = load_config(args.configs_dir / f"{name}.yaml", overlays=args.overlay)
        run_baseline_row(cfg, args.data_root, args.runs_root / cfg.name / "baseline",
                         attribution=cfg.ablation_row == 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
