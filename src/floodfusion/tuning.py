r"""v1.2's equal tuning budget: the same learning-rate grid for each model family, on an inner split of fold 0.

The rule (``reports/v1_2_deviations.md``, fixed before any pooled run):

- **Grid:** :data:`GRID` for every family. 1e-4 is both families' configured default.
- **Inner split** (:func:`inner_split`): fold 0's training groups are divided by whole groups.
  - Whole groups, taken in a seeded order, go to validation while each brings the validation share of windows closer
    to :data:`INNER_VAL_SHARE`; the rest train. Validation, like CV's, scores evaluation-grid chips only.
  - Fold 0's own validation chips fill the test part, which training requires but never scores. Every part stays
    group-disjoint.
- **Choice** (:func:`best_rate`): the rate with the best inner-validation macro IoU (``best_val_macro_iou``). On an
  exact tie, the earlier rate in :data:`GRID`.
**v1.3** (``reports/protocol_v1_3.md`` §3) reuses the rule for fine-tuning: ``--grid 1e-5 3e-5 1e-4``, ``--tag v13``
and ``--init-from-root``, so that each rate's run starts from ``<root>/<run name>/seed42`` (``train --init-from``).

- **Hand-over:** the chosen rate is written as an overlay (``reports/v12_lr_<run name>.yaml``). That family's CV and
  finals pass it after the dataset overlay.

Runs go to ``runs/v12/tune/<run name>/lr<rate>/`` and are resumable. A finished run is reused only if its recorded
configuration hash and split file hash are the ones this rule would produce now; otherwise the stage refuses.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
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


def inner_split(fold0: Any, windows: Mapping[str, str], eval_grid: Collection[str], *, seed: int = 0) -> Any:
    """The inner split of ``fold0`` (a :class:`~floodfusion.splits.SplitAssignment`).

    ``windows`` maps chip -> group; ``eval_grid`` holds the evaluation-grid chips (the only ones validation scores).
    """
    groups = sorted(set(fold0.train_basins))
    if len(groups) < 2:
        raise ValueError("an inner split needs at least two training groups")
    per_group = Counter(windows[c] for c in fold0.train_chips)
    target = INNER_VAL_SHARE * len(fold0.train_chips)
    val: list[str] = []
    total = 0
    for g in (groups[i] for i in np.random.default_rng(seed).permutation(len(groups))):
        if len(val) == len(groups) - 1:
            break
        if abs(total + per_group[g] - target) < abs(total - target):   # closer to the target with g
            val.append(g)
            total += per_group[g]
    val_set = set(val) or {min(groups, key=lambda g: (per_group[g], g))}   # every group overshoots: the smallest
    train_chips = tuple(c for c in fold0.train_chips if windows[c] not in val_set)
    val_chips = tuple(c for c in fold0.train_chips if windows[c] in val_set and c in eval_grid)
    return replace(fold0, train_chips=train_chips, val_chips=val_chips, test_chips=fold0.val_chips,
                   train_basins=tuple(g for g in groups if g not in val_set), val_basins=tuple(sorted(val_set)),
                   test_basins=fold0.val_basins)


def best_rate(results: Mapping[float, float], grid: Sequence[float] = GRID) -> float:
    """The rate with the highest inner-validation IoU; an exact tie goes to the earlier rate of :data:`GRID`."""
    ordered = [r for r in grid if r in results]
    return max(ordered, key=lambda r: (results[r], -ordered.index(r)))


def _yaml_float(rate: float) -> str:
    """``3.0e-05``: YAML reads ``3e-05`` (no decimal point) as a string."""
    return f"{rate:.1e}"


def overlay_text(rate: float, run_name: str, *, tag: str = "v12") -> str:
    label = f"v{tag[1]}.{tag[2:]}" if tag.startswith("v") and len(tag) > 2 else tag
    return (f"# {label} equal tuning budget (floodfusion.tuning): {run_name}'s learning rate, chosen on fold 0's inner "
            f"split.\ntrain:\n  learning_rate: {_yaml_float(rate)}\n")


def _write_inner(chip_dir: Path, cfg: Any) -> None:
    from floodfusion.data.records import atomic_write_bytes, load_manifest, load_manifest_meta
    from floodfusion.splits import compute_norm_stats, freeze_splits, load_splits

    manifest = chip_dir / cfg.data.manifest_name
    sha = load_manifest_meta(manifest)["manifest_sha256"]
    records = {r.chip_id: r for r in load_manifest(manifest)}
    split = inner_split(load_splits(chip_dir / "cv" / "kfold__fold0.json", expected_manifest_sha256=sha),
                        {c: r.basin_id for c, r in records.items()}, {c for c, r in records.items() if r.eval_grid})
    (chip_dir / "tune").mkdir(parents=True, exist_ok=True)
    norm = compute_norm_stats([records[c] for c in split.train_chips], cfg.norm, cfg.grid, seed=0)
    atomic_write_bytes(chip_dir / NORM_FILE, json.dumps({
        "sar_mean_db": list(norm.sar_mean_db), "sar_std_db": list(norm.sar_std_db),
        "computed_on": f"train split of {SPLIT_FILE}", "manifest_sha256": sha}, indent=2).encode("utf-8"))
    freeze_splits(split, chip_dir / SPLIT_FILE, metadata={"inner_of": "cv/kfold__fold0.json",
                                                          "val_share_target": INNER_VAL_SHARE})
    LOGGER.info("inner split: %d train, %d val chips (%d groups)", len(split.train_chips), len(split.val_chips),
                len(split.val_basins))


def _finished(record_path: Path, expected: Mapping[str, str]) -> bool:
    """Whether a finished run of exactly this setting exists; a finished run of another setting is an error."""
    if not record_path.is_file():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "finished":
        return False
    stale = {k: (record.get(k), v) for k, v in expected.items() if record.get(k) != v}
    if stale:
        raise ValueError(f"{record_path.parent} was trained with another setting {stale}; move it away to retrain")
    return True


def tune_row(row: int, data_root: Path, overlays: Sequence[Path], runs_root: Path, *,
             grid: Sequence[float] = GRID, init_from_root: Path | None = None) -> dict[str, Any]:
    """Train ``row`` once per rate of :data:`GRID` on the inner split (resumable); return the rates' results."""
    from floodfusion import train
    from floodfusion.config import config_sha256, config_to_dict, load_config
    from floodfusion.crossval import _config_name

    config = Path("configs") / f"{_config_name(row)}.yaml"
    name = Path(_config_name(row)).name
    results = {}
    init = None if init_from_root is None else Path(init_from_root) / name / f"seed{SEED}"
    for rate in grid:
        run_dir = runs_root / name / f"lr{rate:g}"
        overrides = [f"data.split_file_name={SPLIT_FILE}", f"data.norm_stats_file={NORM_FILE}",
                     "data.split_strategy=leave_one_basin_out", f"train.learning_rate={_yaml_float(rate)}"]
        data = train.load_training_data(load_config(config, overlays=overlays,
                                                    overrides=[*overrides, f"train.seed={SEED}"]), data_root)
        if init is not None:          # a run started from another keeps that run's normalisation
            data = train.adopt_init(data, init)
        # The run records the configuration as training saw it (with the split's norm statistics loaded).
        expected = {"config_sha256": config_sha256(config_to_dict(data.cfg)), "split_sha256": data.split_sha256}
        if not _finished(run_dir / "run.json", expected):
            train.main(["--config", str(config), "--data-root", str(data_root), "--seed", str(SEED),
                        *(["--init-from", str(init)] if init is not None else []),
                        *[a for o in overlays for a in ("--overlay", str(o))], "--run-dir", str(run_dir),
                        *[a for o in overrides for a in ("--set", o)]])
            _finished(run_dir / "run.json", expected)                 # the run just written must match too
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        results[rate] = float(record["best_val_macro_iou"])
        LOGGER.info("%s lr %g: inner-val macro IoU %.4f", name, rate, results[rate])
    return {"run_name": name, "results": {f"{r:g}": v for r, v in results.items()},
            "chosen": best_rate(results, grid), **({"init_from": str(init)} if init is not None else {})}


def main(argv: Sequence[str] | None = None) -> int:
    from floodfusion.config import load_config

    parser = argparse.ArgumentParser(description="Equal learning-rate budget per family on fold 0's inner split.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--runs-root", type=Path, default=Path("runs/v12/tune"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--grid", type=float, nargs="+", default=list(GRID), help="learning rates, in tie order")
    parser.add_argument("--init-from-root", type=Path, default=None,
                        help="v1.3: start each rate's run from <root>/<run name>/seed42 (train --init-from)")
    parser.add_argument("--tag", default="v12", help="report file prefix: <tag>_tuning.json, <tag>_lr_<run>.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(Path("configs/row09_fusion_crossattn.yaml"), overlays=args.overlay)
    chip_dir = args.data_root / Path(cfg.data.chip_root).name
    if not (chip_dir / SPLIT_FILE).is_file():
        _write_inner(chip_dir, cfg)
    summary_path = args.reports / f"{args.tag}_tuning.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["rows"] if summary_path.is_file() else {}
    for row in args.rows:
        result = tune_row(row, args.data_root, args.overlay, args.runs_root, grid=args.grid,
                          init_from_root=args.init_from_root)
        (args.reports / f"{args.tag}_lr_{result['run_name']}.yaml").write_text(
            overlay_text(result["chosen"], result["run_name"], tag=args.tag), encoding="utf-8")
        summary[result["run_name"]] = result
    summary_path.write_text(json.dumps({"grid": list(args.grid), "rows": summary}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
