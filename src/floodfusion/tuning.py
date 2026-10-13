r"""v1.2's equal tuning budget: the same learning-rate grid for each model family, on an inner split of fold 0.

The rule (``reports/v1_2_deviations.md``, fixed before any pooled run):

- **Grid:** :data:`GRID` for every family. 1e-4 is both families' configured default.
- **Inner split** (:func:`inner_split`): fold 0's training groups are divided by whole groups.
  - About :data:`INNER_VAL_SHARE` of their windows (seeded) go to validation; the rest train.
  - Fold 0's own validation chips fill the test part, which training requires but never scores. Every part stays
    group-disjoint.
- **Choice** (:func:`best_rate`): the rate with the best inner-validation macro IoU (``best_val_macro_iou``). On an
  exact tie, the earlier rate in :data:`GRID`.
- **Hand-over:** the chosen rate is written as an overlay (``reports/v12_lr_<run name>.yaml``). That family's CV and
  finals pass it after the dataset overlay.

Runs go to ``runs/v12/tune/<run name>/lr<rate>/`` and are resumable: a finished run is not trained again.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["GRID", "INNER_VAL_SHARE", "best_rate", "inner_split", "main", "overlay_text"]

LOGGER = logging.getLogger(__name__)

GRID = (3e-5, 1e-4, 3e-4)
INNER_VAL_SHARE = 0.2
SEED = 42
SPLIT_FILE, NORM_FILE = "tune/inner.json", "tune/inner.norm_stats.json"


def inner_split(fold0: Any, windows: Mapping[str, str], *, seed: int = 0) -> Any:
    """The inner split of ``fold0`` (a :class:`~floodfusion.splits.SplitAssignment`); ``windows`` maps chip -> group."""
    groups = sorted(set(fold0.train_basins))
    if len(groups) < 2:
        raise ValueError("an inner split needs at least two training groups")
    per_group = Counter(windows[c] for c in fold0.train_chips)
    target = INNER_VAL_SHARE * len(fold0.train_chips)
    val: list[str] = []
    for g in (groups[i] for i in np.random.default_rng(seed).permutation(len(groups))):
        if sum(per_group[v] for v in val) >= target or len(val) == len(groups) - 1:
            break
        val.append(g)
    val_set = set(val)
    train_chips = tuple(c for c in fold0.train_chips if windows[c] not in val_set)
    val_chips = tuple(c for c in fold0.train_chips if windows[c] in val_set)
    return replace(fold0, train_chips=train_chips, val_chips=val_chips, test_chips=fold0.val_chips,
                   train_basins=tuple(g for g in groups if g not in val_set), val_basins=tuple(sorted(val_set)),
                   test_basins=fold0.val_basins)


def best_rate(results: Mapping[float, float]) -> float:
    """The rate with the highest inner-validation IoU; an exact tie goes to the earlier rate of :data:`GRID`."""
    ordered = [r for r in GRID if r in results]
    return max(ordered, key=lambda r: (results[r], -ordered.index(r)))


def _yaml_float(rate: float) -> str:
    """``3.0e-05``: YAML reads ``3e-05`` (no decimal point) as a string."""
    return f"{rate:.1e}"


def overlay_text(rate: float, run_name: str) -> str:
    return (f"# v1.2 equal tuning budget (floodfusion.tuning): {run_name}'s learning rate, chosen on fold 0's inner "
            f"split.\ntrain:\n  learning_rate: {_yaml_float(rate)}\n")


def _write_inner(chip_dir: Path, cfg: Any) -> None:
    from floodfusion.data.records import atomic_write_bytes, load_manifest, load_manifest_meta
    from floodfusion.splits import compute_norm_stats, freeze_splits, load_splits

    manifest = chip_dir / cfg.data.manifest_name
    sha = load_manifest_meta(manifest)["manifest_sha256"]
    records = {r.chip_id: r for r in load_manifest(manifest)}
    split = inner_split(load_splits(chip_dir / "cv" / "kfold__fold0.json", expected_manifest_sha256=sha),
                        {c: r.basin_id for c, r in records.items()})
    (chip_dir / "tune").mkdir(parents=True, exist_ok=True)
    norm = compute_norm_stats([records[c] for c in split.train_chips], cfg.norm, cfg.grid, seed=0)
    atomic_write_bytes(chip_dir / NORM_FILE, json.dumps({
        "sar_mean_db": list(norm.sar_mean_db), "sar_std_db": list(norm.sar_std_db),
        "computed_on": f"train split of {SPLIT_FILE}", "manifest_sha256": sha}, indent=2).encode("utf-8"))
    freeze_splits(split, chip_dir / SPLIT_FILE, metadata={"inner_of": "cv/kfold__fold0.json",
                                                          "val_share_target": INNER_VAL_SHARE})
    LOGGER.info("inner split: %d train, %d val chips (%d groups)", len(split.train_chips), len(split.val_chips),
                len(split.val_basins))


def tune_row(row: int, data_root: Path, overlays: Sequence[Path], runs_root: Path) -> dict[str, Any]:
    """Train ``row`` once per rate of :data:`GRID` on the inner split (resumable); return the rates' results."""
    from floodfusion import train
    from floodfusion.crossval import _config_name

    config = Path("configs") / f"{_config_name(row)}.yaml"
    name = Path(_config_name(row)).name
    results = {}
    for rate in GRID:
        run_dir = runs_root / name / f"lr{rate:g}"
        record = run_dir / "run.json"
        done = record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("status") == "finished"
        if not done:
            train.main(["--config", str(config), "--data-root", str(data_root), "--seed", str(SEED),
                        *[a for o in overlays for a in ("--overlay", str(o))], "--run-dir", str(run_dir),
                        "--set", f"data.split_file_name={SPLIT_FILE}", "--set", f"data.norm_stats_file={NORM_FILE}",
                        "--set", "data.split_strategy=leave_one_basin_out", "--set", f"train.learning_rate={_yaml_float(rate)}"])
        results[rate] = float(json.loads(record.read_text(encoding="utf-8"))["best_val_macro_iou"])
        LOGGER.info("%s lr %g: inner-val macro IoU %.4f", name, rate, results[rate])
    return {"run_name": name, "results": {f"{r:g}": v for r, v in results.items()}, "chosen": best_rate(results)}


def main(argv: Sequence[str] | None = None) -> int:
    from floodfusion.config import load_config

    parser = argparse.ArgumentParser(description="Equal learning-rate budget per family on fold 0's inner split.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--runs-root", type=Path, default=Path("runs/v12/tune"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path("configs/row09_fusion_crossattn.yaml"), overlays=args.overlay)
    chip_dir = args.data_root / Path(cfg.data.chip_root).name
    if not (chip_dir / SPLIT_FILE).is_file():
        _write_inner(chip_dir, cfg)
    summary_path = args.reports / "v12_tuning.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["rows"] if summary_path.is_file() else {}
    for row in args.rows:
        result = tune_row(row, args.data_root, args.overlay, args.runs_root)
        (args.reports / f"v12_lr_{result['run_name']}.yaml").write_text(
            overlay_text(result["chosen"], result["run_name"]), encoding="utf-8")
        summary[result["run_name"]] = result
    summary_path.write_text(json.dumps({"grid": list(GRID), "rows": summary}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
