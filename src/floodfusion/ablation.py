r"""``P1-M5``: run the whole ablation grid and write the table — one command.

Run as (``make ablation``)::

    python -m floodfusion.ablation --data-root data/synthetic --runs-root runs \
        --overlay configs/overlays/synthetic_demo.yaml

Order of work:

1. Rows 1--3: the physics baseline (:mod:`floodfusion.baseline_runner`), CPU.
2. Row 9, first seed — timed. The **seed protocol** is then fixed from that measurement,
   before any other learned row runs: if the projected grid fits ``--budget-minutes``, every
   learned row gets every seed; otherwise the affordable protocol of ``docs/09`` section 2.2
   (row 9 with every seed, the other rows with the first). The decision and its inputs are
   written to ``<runs_root>/ablation_protocol.json`` and stated in the table.
3. Rows 4--8, 10, 12, 13 and the remaining row-9 seeds, each trained by
   :func:`floodfusion.train.fit` and evaluated by :func:`floodfusion.evaluate.evaluate_run`.
4. Row 11 = every row-9 checkpoint evaluated SAR-only; supplementary: row 10 SAR-only (what
   *removing* modality dropout costs when optical is absent).
5. The table (:func:`floodfusion.reporting.build_ablation_table`).

Finished runs (``run.json`` status ``finished`` and an evaluation on disk) are skipped, so an
interrupted grid resumes where it stopped. Latency is measured once per (architecture,
SAR-only) pair — rows sharing an architecture share the same network cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from floodfusion.config import ExperimentConfig, load_config

__all__ = ["LEARNED_ROWS", "ROW_CONFIGS", "main", "run_grid"]

LOGGER = logging.getLogger(__name__)

ROW_CONFIGS = {
    1: "row01_otsu_global", 2: "row02_baseline_adaptive", 3: "row03_baseline_mndwi",
    4: "row04_unet_sar_only", 5: "row05_unet_sar_dem", 6: "row06_unet_optical_only",
    7: "row07_fusion_early", 8: "row08_fusion_late", 9: "row09_fusion_crossattn",
    10: "row10_fusion_no_modality_dropout", 11: "row11_fusion_eval_sar_only",
    12: "row12_fusion_bce_loss", 13: "row13_fusion_random_split",
}
#: Learned rows in execution order (row 9 first: it calibrates the time budget).
LEARNED_ROWS = (9, 4, 5, 6, 7, 8, 10, 12, 13)


class _LatencyCache:
    """Measure inference cost once per (architecture, SAR-only); reuse it for every row."""

    def __init__(self, warmup_s: float, throughput_s: float) -> None:
        self.warmup_s, self.throughput_s = warmup_s, throughput_s
        self.cache: dict[tuple[str, bool], dict[str, Any]] = {}

    def __call__(self, model: Any, device: Any, sar_only: bool) -> dict[str, Any]:
        from floodfusion.utils.latency import measure_inference

        cfg = model.cfg
        key = (f"{cfg.architecture}{'+terrain' if cfg.use_terrain else ''}", sar_only)
        if key not in self.cache:
            self.cache[key] = measure_inference(model, device, sar_only=sar_only, warmup_s=self.warmup_s,
                                                throughput_s=self.throughput_s)
        return {**self.cache[key], "shared_measurement": key[0]}


def _finished(run_dir: Path) -> bool:
    record = run_dir / "run.json"
    if not record.is_file() or not (run_dir / "eval_test.json").is_file():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("status") == "finished"


def _config(configs_dir: Path, row: int, overlays: Sequence[Path], seed: int | None = None) -> ExperimentConfig:
    overrides = [f"train.seed={seed}"] if seed is not None else []
    return load_config(configs_dir / f"{ROW_CONFIGS[row]}.yaml", overlays=overlays, overrides=overrides)


def _train_and_evaluate(row: int, seed: int, args: argparse.Namespace, device: Any, latency: _LatencyCache,
                        overlays: Sequence[Path]) -> float:
    """Train + evaluate one (row, seed) unless already finished; returns wall seconds spent."""
    from floodfusion.evaluate import evaluate_run
    from floodfusion.runs import run_dir_for
    from floodfusion.train import EmptySplitError, fit, load_training_data

    cfg = _config(args.configs_dir, row, overlays, seed)
    run_dir = run_dir_for(args.runs_root, cfg.name, seed)
    if _finished(run_dir):
        LOGGER.info("row %d seed %d already finished — skipped", row, seed)
        return 0.0
    started = time.perf_counter()
    try:
        data = load_training_data(cfg, args.data_root)
    except EmptySplitError as exc:
        # Row 6 on data without cloud-free chips in some split: the row is "not run", the grid goes on.
        LOGGER.warning("row %d seed %d not run: %s", row, seed, exc)
        return 0.0
    fit(data, run_dir, device=device, provenance={
        "config_path": str(args.configs_dir / f"{ROW_CONFIGS[row]}.yaml"), "overlays": [str(o) for o in overlays],
        "overrides": [f"train.seed={seed}"], "data_root": str(args.data_root), "driver": "floodfusion.ablation",
    })
    evaluate_run(run_dir, data_root=args.data_root, device=device, latency=latency)
    return time.perf_counter() - started


def _evaluate_sar_only(row: int, seed: int, args: argparse.Namespace, device: Any, latency: _LatencyCache,
                       overlays: Sequence[Path]) -> None:
    from floodfusion.evaluate import evaluate_run
    from floodfusion.runs import run_dir_for

    source = run_dir_for(args.runs_root, _config(args.configs_dir, row, overlays, seed).name, seed)
    name = ROW_CONFIGS[11] if row == 9 else f"{ROW_CONFIGS[row]}__eval_sar_only"
    out = run_dir_for(args.runs_root, name, seed)
    if _finished(out) or not (source / "best.pt").is_file():
        return
    evaluate_run(source, data_root=args.data_root, device=device, force_sar_only=True, out_dir=out, latency=latency)


#: Distinct (architecture, SAR-only) latency measurements the grid makes: 6 networks + 2 SAR-only.
_N_LATENCY_MEASUREMENTS = 8


def _decide_protocol(first_run_s: float, latency_s: float, seeds: Sequence[int], budget_min: float,
                     requested: str) -> dict[str, Any]:
    """All seeds for every learned row if the projected grid fits the budget, else docs/09 2.2.

    The projection is deliberately pessimistic: every learned row is assumed to cost as much
    as the reference row 9 (the largest network); latency measurements are counted once each.
    """
    n_rows = len(LEARNED_ROWS)
    per_run = max(first_run_s - latency_s, 0.0)
    projected_full = (per_run * n_rows * len(seeds) + _N_LATENCY_MEASUREMENTS * latency_s) / 60.0
    if requested == "full" or (requested == "auto" and projected_full <= budget_min):
        chosen = "full"
    else:
        chosen = "affordable"
    return {"protocol": chosen, "requested": requested, "first_run_minutes": first_run_s / 60.0,
            "projected_full_grid_minutes": projected_full, "budget_minutes": budget_min, "seeds": list(seeds),
            "rule": "every learned row x every seed if the projected full grid fits the budget; otherwise "
                    "docs/09 section 2.2: row 9 x every seed, other rows x first seed"}


def run_grid(args: argparse.Namespace) -> Path:
    from floodfusion.baseline_runner import run_baseline_row
    from floodfusion.reporting import build_ablation_table
    from floodfusion.runs import write_json
    from floodfusion.train import resolve_device

    device = resolve_device(args.device)
    overlays = list(args.overlay)
    latency = _LatencyCache(args.warmup_s, args.throughput_s)
    for row in (1, 2, 3):
        cfg = _config(args.configs_dir, row, overlays)
        out = args.runs_root / cfg.name / "baseline"
        if not _finished(out):
            run_baseline_row(cfg, args.data_root, out, attribution=row == 2)
    seeds = list(args.seeds)
    first = _train_and_evaluate(9, seeds[0], args, device, latency, overlays)
    protocol_path = args.runs_root / "ablation_protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))  # resume: never re-decide
    else:
        if first == 0.0:  # resumed after the first run but before the decision was written
            first = float(json.loads((args.runs_root / ROW_CONFIGS[9] / f"seed{seeds[0]}" / "run.json")
                                     .read_text(encoding="utf-8"))["train_seconds"])
        latency_s = args.warmup_s + args.throughput_s + 5.0 if device.type == "cuda" else 0.0
        protocol = _decide_protocol(first, latency_s, seeds, args.budget_minutes, args.protocol)
        write_json(protocol_path, protocol)
    LOGGER.info("seed protocol: %s (projected full grid %.0f min, budget %.0f min)", protocol["protocol"],
                protocol["projected_full_grid_minutes"], protocol["budget_minutes"])
    plan = [(row, seed) for row in LEARNED_ROWS for seed in seeds
            if (row, seed) != (9, seeds[0]) and (protocol["protocol"] == "full" or row == 9 or seed == seeds[0])]
    spent, started = [], time.perf_counter()
    for i, (row, seed) in enumerate(plan, start=1):
        spent.append(_train_and_evaluate(row, seed, args, device, latency, overlays))
        done = [s for s in spent if s > 0]
        eta = (sum(done) / len(done)) * (len(plan) - i) / 60.0 if done else math.nan
        LOGGER.info("grid progress %d/%d (%.0f min elapsed, ETA %.0f min)", i, len(plan),
                    (time.perf_counter() - started) / 60.0, eta)
    for seed in seeds:
        _evaluate_sar_only(9, seed, args, device, latency, overlays)
        _evaluate_sar_only(10, seed, args, device, latency, overlays)
    build_ablation_table(args.runs_root, out=args.out, allow_dirty=args.allow_dirty)
    return args.out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the P1-M5 ablation grid end to end.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--protocol", choices=("auto", "full", "affordable"), default="auto")
    parser.add_argument("--budget-minutes", type=float, default=75.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--throughput-s", type=float, default=60.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/ablation_table.md"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_grid(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
