r"""Evaluation and ablation-table generation.

Milestones ``P1-M5`` and ``P1-M6``. Every row of the ablation table regenerates by one
command from a recorded git SHA, so this module is the single entry point for every
number in the report.

Run as::

    python -m floodfusion.evaluate --run-dir runs/row09_fusion_crossattn/seed42 --split test
    python -m floodfusion.evaluate --run-dir runs/row09_fusion_crossattn/seed42 --force-sar-only \
        --out-dir runs/row11_fusion_eval_sar_only/seed42
    python -m floodfusion.evaluate --ablation-table --runs-root runs/ --out reports/ablation_table.md

The sequence (``docs/09`` sections 2.3 and 4.5), identical for every learned row:

1. Load the best checkpoint and the frozen split; **assert** the manifest hash, the split
   binding and the normalisation statistics are the ones the run trained on.
2. **Validation** pass: collect a uniform reservoir of valid-pixel logits; apply the prior
   correction (``docs/05`` 8.3) and fit a temperature (``docs/09`` 4.3); report ECE (quantile
   bins) before and after.
3. **Test** pass, once: per-basin confusion matrices at a fixed threshold grid on the
   calibrated scale, predicted with 64 px of context and scored on the central chip;
   natural class balance; invalid pixels excluded.
4. Report IoU/F1 at :math:`\tau^\star = 1/(1+r)` **and** at the best achievable threshold
   (an oracle, labelled as such), the threshold selected on validation, raw-score
   :math:`\sigma(z) > \tau^\star` (what skipping calibration would give), AP, macro and pooled
   basin-bootstrap CIs, ECE on test, the fraction of pixels excluded per validity flag, and —
   on the GPU — latency per ``docs/09`` section 7.

References
----------
``docs/09_evaluation_ablation_and_report.md``.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.calibration import (
    LogitReservoir,
    apply_calibration,
    calibrate_logits,
    fit_temperature,
)
from floodfusion.config import (
    ExperimentConfig,
    ValidityFlag,
    config_from_dict,
    config_sha256,
    config_to_dict,
)
from floodfusion.metrics import (
    ConfusionMatrix,
    average_precision,
    basin_bootstrap_ci,
    expected_calibration_error,
    macro_iou,
    precision_recall_curve,
    sum_confusions,
)
from floodfusion.runs import load_run_record, write_json, write_run_record

__all__ = ["build_ablation_table", "evaluate_run", "fit_temperature", "main"]

LOGGER = logging.getLogger(__name__)

RESERVOIR_CAPACITY = 2_000_000


def build_ablation_table(*args: Any, **kwargs: Any) -> str:
    """Assemble the 13-row ablation table of ``MILESTONES.md`` section 6.

    Implemented in :func:`floodfusion.reporting.build_ablation_table`; re-exported here so
    ``python -m floodfusion.evaluate --ablation-table`` stays the one command.
    """
    from floodfusion.reporting import build_ablation_table as _build

    return _build(*args, **kwargs)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cm_dict(cm: ConfusionMatrix) -> dict[str, int]:
    return {"tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn}


def _basin_summary(per_basin: Mapping[str, ConfusionMatrix]) -> dict[str, dict[str, float]]:
    return {b: {"iou": m.iou, "f1": m.f1, "precision": m.precision, "recall": m.recall,
                "n_valid": m.tp + m.fp + m.fn + m.tn, "n_flood": m.tp + m.fn, **_cm_dict(m)}
            for b, m in sorted(per_basin.items())}


def _macro(per_basin: Mapping[str, ConfusionMatrix], attr: str) -> float:
    values = [getattr(m, attr) for m in per_basin.values()]
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _load(run_dir: Path, data_root: Path | None, device: Any, force_sar_only: bool) -> tuple[Any, ...]:
    """Config, data, model and record of a finished run, with every binding re-checked."""
    import torch

    from floodfusion.models.zoo import build_model
    from floodfusion.train import load_training_data

    record = load_run_record(run_dir)
    if record.get("status") != "finished":
        raise ValueError(f"{run_dir}: run status is {record.get('status')!r}, not 'finished'")
    cfg = config_from_dict(record["config"])
    if force_sar_only:
        cfg = replace(cfg, eval=replace(cfg.eval, force_sar_only=True))
    data = load_training_data(cfg, Path(data_root or record["data_root"]))
    if data.manifest_sha256 != record["manifest_sha256"]:
        raise ValueError(f"{run_dir}: trained on manifest {record['manifest_sha256'][:12]}…, "
                         f"data root now holds {data.manifest_sha256[:12]}…")
    if data.cfg.norm != cfg.norm:
        raise ValueError(f"{run_dir}: norm_stats.json changed since training")
    state = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    if state["config_sha256"] != record["config_sha256"]:
        raise ValueError(f"{run_dir}: best.pt belongs to a different configuration")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(state["model"])
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return cfg, data, model, record, int(state["epoch"])


def _loader(records: Sequence[Any], cfg: ExperimentConfig) -> Any:
    from floodfusion.data.dataset import FloodChipDataset, build_dataloader
    from floodfusion.train import EVAL_CONTEXT_PX

    ds = FloodChipDataset(records, cfg.norm, cfg.data, cfg.grid, context_px=EVAL_CONTEXT_PX)
    return build_dataloader(ds, cfg.data, batch_size=cfg.train.micro_batch_size, shuffle=False, drop_last=False)


def flag_fractions(records: Sequence[Any], cfg: ExperimentConfig) -> dict[str, float]:
    """Fraction of the scored pixels carrying each validity flag (``docs/03`` section 5.4)."""
    import rasterio
    from rasterio.windows import Window

    from floodfusion.data.encoding import ChipBand

    counts = dict.fromkeys((f.name for f in ValidityFlag), 0)
    total = 0
    by_path: dict[str, list[Any]] = {}
    for r in records:
        by_path.setdefault(str(r.raster_path), []).append(r)
    for path, recs in by_path.items():
        if not Path(path).is_file():
            continue
        with rasterio.open(path) as src:
            for r in recs:
                col, row, w, h = r.window
                v = src.read(int(ChipBand.VALIDITY) + 1, window=Window(col, row, w, h)).astype(np.int64)
                total += v.size
                for flag in ValidityFlag:
                    counts[flag.name] += int(((v & int(flag)) > 0).sum())
    return {k: (c / total if total else float("nan")) for k, c in counts.items()}


def _thresholds(cfg: ExperimentConfig, temperature: float, offset: float, val_raw_best: float) -> tuple[tuple[float, ...], dict[str, float]]:
    """The calibrated-scale grid plus the special points, each exactly on the grid."""
    tau = cfg.eval.decision_threshold
    special = {
        "tau_star": tau,
        # sigma(z) > tau without calibration, expressed on the calibrated scale:
        "raw_tau_star": _sigmoid((_logit(tau) + offset) / temperature),
        # the best validation threshold (raw scale during training), on the calibrated scale:
        "val_selected": _sigmoid((_logit(val_raw_best) + offset) / temperature),
    }
    special = {k: min(max(v, 1e-9), 1.0 - 1e-9) for k, v in special.items()}  # strictly inside (0, 1)
    grid = sorted(set(cfg.eval.pr_thresholds) | set(special.values()))
    return tuple(grid), special


def _test_metrics(test: Mapping[str, Any], special: Mapping[str, float], cfg: ExperimentConfig,
                  seed: int) -> dict[str, Any]:
    thresholds = list(test["thresholds"])
    per_basin_all: Mapping[str, Sequence[ConfusionMatrix]] = test["per_basin"]

    def at(t: float) -> dict[str, ConfusionMatrix]:
        k = thresholds.index(t)
        return {b: v[k] for b, v in per_basin_all.items()}

    tau_cm = at(special["tau_star"])
    pooled = [sum_confusions([v[k] for v in per_basin_all.values()]) for k in range(len(thresholds))]
    macro_curve = [macro_iou({b: v[k] for b, v in per_basin_all.items()}) for k in range(len(thresholds))]
    oracle = int(np.nanargmax(np.where(np.isfinite(macro_curve), macro_curve, -1.0)))
    precisions, recalls = precision_recall_curve(tuple(pooled))
    n_boot, alpha = cfg.eval.n_bootstrap, cfg.eval.bootstrap_alpha
    macro_ci = basin_bootstrap_ci(dict(tau_cm), n_boot=n_boot, alpha=alpha, seed=seed, statistic="macro")
    pooled_ci = basin_bootstrap_ci(dict(tau_cm), n_boot=n_boot, alpha=alpha, seed=seed, statistic="pooled")
    pooled_tau = sum_confusions(list(tau_cm.values()))
    return {
        "tau_star": special["tau_star"],
        "macro_iou": macro_iou(tau_cm), "macro_iou_ci": list(macro_ci[1:]),
        "macro_f1": _macro(tau_cm, "f1"), "macro_precision": _macro(tau_cm, "precision"),
        "macro_recall": _macro(tau_cm, "recall"),
        "pooled_iou": pooled_tau.iou, "pooled_iou_ci": list(pooled_ci[1:]), "pooled_f1": pooled_tau.f1,
        "per_basin": _basin_summary(tau_cm),
        "macro_iou_raw_tau_star": macro_iou(at(special["raw_tau_star"])),
        "macro_iou_val_selected": macro_iou(at(special["val_selected"])),
        "val_selected_threshold": special["val_selected"],
        "macro_iou_oracle": macro_curve[oracle], "oracle_threshold": thresholds[oracle],
        "average_precision": average_precision(precisions, recalls),
        "pr_curve": {"precision": list(precisions), "recall": list(recalls)},
        "operating_point": {"precision": pooled_tau.precision, "recall": pooled_tau.recall},
        "per_basin_confusion_by_threshold": {b: [[m.tp, m.fp, m.fn, m.tn] for m in v]
                                             for b, v in per_basin_all.items()},
        "thresholds": thresholds,
    }


def evaluate_run(
    run_dir: Path | str,
    *,
    split: str = "test",
    data_root: Path | str | None = None,
    device: Any = None,
    force_sar_only: bool = False,
    out_dir: Path | str | None = None,
    latency: Callable[[Any, Any, bool], dict[str, Any]] | None = None,
    figures: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    r"""Evaluate one trained run on ``split``, per basin; writes ``eval_<split>.json``.

    ``force_sar_only`` evaluates the same checkpoint with the optical branch replaced by
    the missing-modality embedding everywhere (ablation row 11); calibration is then refit
    on SAR-only validation logits, because the SAR-only predictor is a different model of
    the same weights. ``latency`` is called as ``latency(model, device, sar_only)``.
    """
    from floodfusion.train import EVAL_CONTEXT_PX, resolve_device, validate

    run_dir = Path(run_dir)
    device = resolve_device(device) if not hasattr(device, "type") else device
    cfg, data, model, record, best_epoch = _load(run_dir, Path(data_root) if data_root else None, device,
                                                 force_sar_only)
    sar_only = cfg.eval.force_sar_only
    val_reservoir = LogitReservoir(RESERVOIR_CAPACITY, seed=seed)
    val = validate(model, _loader(data.val, cfg), device=device, cfg=cfg, force_sar_only=sar_only,
                   crop_px=EVAL_CONTEXT_PX, logit_sink=val_reservoir)
    z_val, y_val = val_reservoir.arrays()
    calibration = calibrate_logits(z_val, y_val, pi_train=record["pi_train"], pi_deploy=record["pi_deploy"])
    thresholds, special = _thresholds(cfg, calibration.temperature, calibration.offset, val["best_threshold"])
    target = data.test if split == "test" else data.val
    test_reservoir = LogitReservoir(RESERVOIR_CAPACITY, seed=seed + 1)
    scored = validate(model, _loader(target, cfg), device=device, cfg=cfg, thresholds=thresholds,
                      force_sar_only=sar_only, crop_px=EVAL_CONTEXT_PX, logit_sink=test_reservoir,
                      temperature=calibration.temperature, offset=calibration.offset)
    metrics = _test_metrics(scored, special, cfg, seed)
    z_t, y_t = test_reservoir.arrays()
    ece_raw, _ = expected_calibration_error(apply_calibration(z_t, 1.0), y_t)
    ece_cal, _ = expected_calibration_error(apply_calibration(z_t, calibration.temperature, calibration.offset), y_t)
    result = {
        "run_dir": str(run_dir), "split": split, "name": cfg.name if not sar_only else _row11_name(cfg),
        # Row 11 is row 9 evaluated SAR-only; any other SAR-only evaluation is supplementary (no row).
        "ablation_row": (11 if cfg.ablation_row == 9 else None) if sar_only else cfg.ablation_row,
        "force_sar_only": sar_only, "best_epoch": best_epoch, "n_chips": len(target),
        "calibration": calibration.summary(), "reliability_val": {
            "raw": calibration.reliability_raw, "calibrated": calibration.reliability_calibrated},
        f"ece_{split}_raw": ece_raw, f"ece_{split}_calibrated": ece_cal, **metrics,
        "flag_fractions": flag_fractions(target, cfg),
        "val_best_macro_iou_raw": val["best_macro_iou"], "manifest_sha256": data.manifest_sha256,
        "synthetic_data": data.is_synthetic,
    }
    if latency is not None and getattr(device, "type", "") == "cuda":
        result["latency"] = latency(model, device, sar_only)
    out = Path(out_dir) if out_dir else run_dir
    if sar_only and out != run_dir:
        _write_derived_record(out, record, cfg, run_dir)
    write_json(out / f"eval_{split}.json", result)
    if figures:
        _figures(result, out, cfg)
    LOGGER.info("%s [%s]: macro IoU %.3f [%.3f, %.3f] at tau*=%.2f (raw-score %.3f, oracle %.3f), AP %.3f, "
                "T=%.2f, ECE %.4f -> %.4f", result["name"], split, result["macro_iou"], *result["macro_iou_ci"],
                result["tau_star"], result["macro_iou_raw_tau_star"], result["macro_iou_oracle"],
                result["average_precision"], calibration.temperature, ece_raw, ece_cal)
    return result


def _row11_name(cfg: ExperimentConfig) -> str:
    return "row11_fusion_eval_sar_only" if cfg.ablation_row == 9 else f"{cfg.name}__eval_sar_only"


def _write_derived_record(out: Path, source: Mapping[str, Any], cfg: ExperimentConfig, source_dir: Path) -> None:
    """A run record for an evaluation-only row (row 11): no training, weights from ``source_dir``."""
    derived = {**source, "name": _row11_name(cfg), "ablation_row": 11 if cfg.ablation_row == 9 else None,
               "config": config_to_dict(cfg), "config_sha256": config_sha256(config_to_dict(cfg)),
               "source_run": str(source_dir), "trained": False}
    write_run_record(out, derived)


def _figures(result: Mapping[str, Any], out: Path, cfg: ExperimentConfig) -> None:
    from floodfusion.figures import plot_pr_curve, plot_reliability

    split = result["split"]
    plot_pr_curve(result["pr_curve"]["precision"], result["pr_curve"]["recall"],
                  (result["operating_point"]["precision"], result["operating_point"]["recall"]),
                  out / "figures" / f"pr_{split}.png", title=f"{result['name']} — {split} (pooled over basins)",
                  cost_ratio=cfg.eval.cost_ratio, tau=result["tau_star"], ap=result["average_precision"])
    plot_reliability(result["reliability_val"]["raw"], result["reliability_val"]["calibrated"],
                     out / "figures" / "reliability_val.png", title=f"{result['name']} — validation",
                     ece_before=result["calibration"]["ece_raw"], ece_after=result["calibration"]["ece_calibrated"])


def _latency_fn(warmup_s: float, throughput_s: float) -> Callable[[Any, Any, bool], dict[str, Any]]:
    from floodfusion.utils.latency import measure_inference

    def run(model: Any, device: Any, sar_only: bool) -> dict[str, Any]:
        return measure_inference(model, device, sar_only=sar_only, warmup_s=warmup_s, throughput_s=throughput_s)

    return run


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Project 01 runs.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", type=Path, default=None, help="defaults to the run's recorded data root")
    parser.add_argument("--force-sar-only", action="store_true", help="ablation row 11: optical replaced everywhere")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latency", action="store_true", help="measure latency/throughput (GPU, docs/09 7)")
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--throughput-s", type=float, default=60.0)
    parser.add_argument("--ablation-table", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/ablation_table.md"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.ablation_table:
        if args.runs_root is None:
            parser.error("--ablation-table needs --runs-root")
        build_ablation_table(args.runs_root, out=args.out, allow_dirty=args.allow_dirty)
        return 0
    if args.run_dir is None:
        parser.error("give --run-dir (or --ablation-table --runs-root)")
    evaluate_run(args.run_dir, split=args.split, data_root=args.data_root, device=args.device,
                 force_sar_only=args.force_sar_only, out_dir=args.out_dir,
                 latency=_latency_fn(args.warmup_s, args.throughput_s) if args.latency else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
