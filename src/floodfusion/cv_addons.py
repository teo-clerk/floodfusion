r"""v1.2: are the inference add-ons adopted? Scored on whole CV test stacks (``reports/protocol_v1_2.md`` §3).

The add-ons operate on a whole scene, so chip-level CV metrics cannot judge them. Here every test stack of every
fold is predicted whole (:func:`floodfusion.predict.predict_stack`, the operator the CEMS benchmark and the blind test
use). Its scores are kept as histograms (:mod:`floodfusion.histograms`), so every variant is evaluated from one
prediction:

- **plain:** IoU at the run's τ*;
- **self-calibration:** τ* moved by the scene's bounded EM prior shift, fitted on its own unlabelled scores;
- **HAND gate:** no flood where HAND ≥ :data:`HAND_MAX_M` (the physics baselines' hard cut);
- **both.**

**The rule** (``reports/v1_2_deviations.md``, fixed before any pooled-CV run):
- confusion counts are pooled per split group, and IoU is averaged over groups;
- an add-on is adopted if it raises that mean by at least :data:`MIN_GAIN` **and** raises the IoU of more than half of
  the groups.

**Scored pixels** are those the chip metrics score (:func:`floodfusion.data.encoding.normalise`): no validity flag,
finite SAR, terrain and label.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion import histograms

__all__ = ["ARMS", "HAND_MAX_M", "MIN_GAIN", "adoption", "fold_stacks", "main", "scene_layers", "score_runs",
           "stack_entry"]

LOGGER = logging.getLogger(__name__)

HAND_MAX_M = 15.0
MIN_GAIN = 0.005
ARMS = ("plain", "selfcal", "hand", "selfcal+hand")


def scene_layers(raw: np.ndarray, grid_cfg: Any, norm: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(scored, label, hand_m)`` of a stored ``(16, H, W)`` stack: the pixels chip metrics score, their label, HAND."""
    from floodfusion.data.encoding import decode_physical

    chip = decode_physical(raw, grid_cfg, norm)
    sar_ok = np.isfinite(chip.gamma0_db).all(axis=0)
    terrain_ok = np.isfinite(chip.slope_deg) & np.isfinite(chip.log1p_hand) & np.isfinite(chip.local_incidence_deg)
    label_ok = np.isfinite(chip.label)
    scored = (chip.validity == 0) & sar_ok & terrain_ok & label_ok
    return scored, np.where(label_ok, chip.label > 0.5, False), np.expm1(chip.log1p_hand)


def stack_entry(prob: np.ndarray, scored: np.ndarray, label: np.ndarray, hand_m: np.ndarray) -> dict[str, Any]:
    """Histograms of one predicted stack: ``plain`` and ``hand``-gated, each ``pos`` / ``neg`` (scored) and ``all``."""
    gated = np.where(hand_m >= HAND_MAX_M, 0.0, prob)          # NaN HAND: no evidence, the score is kept
    have = scored & np.isfinite(prob)
    out = {}
    for arm, p in (("plain", prob), ("hand", gated)):
        out[arm] = {"pos": histograms.histogram(p, have & label).tolist(),
                    "neg": histograms.histogram(p, have & ~label).tolist(),
                    "all": histograms.histogram(p, np.isfinite(p)).tolist()}
    return out


def _counts(entry: Mapping[str, Any], arm: str, tau: float, pi0: float) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` of one stack under one arm."""
    hist = entry["hand" if "hand" in arm else "plain"]
    pos, neg = np.asarray(hist["pos"]), np.asarray(hist["neg"])
    threshold = tau
    if "selfcal" in arm:
        threshold = histograms.shifted_threshold(tau, histograms.em_prior_shift(np.asarray(hist["all"]), pi0))
    return histograms.confusion_above(pos, neg, threshold)[:3]


def adoption(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Per-arm macro IoU over groups, each add-on's gain and the groups it helps, and whether it is adopted."""
    groups: dict[str, dict[str, list[int]]] = {}
    for run in scores["runs"].values():
        for entry in run["stacks"].values():
            for arm in ARMS:
                tp, fp, fn = _counts(entry, arm, run["tau"], run["pi0"])
                acc = groups.setdefault(entry["group"], {}).setdefault(arm, [0, 0, 0])
                acc[0], acc[1], acc[2] = acc[0] + tp, acc[1] + fp, acc[2] + fn
    iou = {g: {arm: (c[0] / sum(c) if sum(c) else math.nan) for arm, c in arms.items()} for g, arms in groups.items()}
    scored = sorted(g for g, v in iou.items() if not math.isnan(v["plain"]))
    macro = {arm: float(np.mean([iou[g][arm] for g in scored])) if scored else math.nan for arm in ARMS}
    result: dict[str, Any] = {"n_groups": len(scored), "macro_iou": macro, "per_group": iou}
    for arm in ARMS[1:]:
        helped = sum(iou[g][arm] > iou[g]["plain"] for g in scored)
        gain = macro[arm] - macro["plain"]
        result[arm] = {"gain": gain, "groups_helped": helped,
                       "adopted": bool(gain >= MIN_GAIN and helped > len(scored) / 2)}
    return result


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def fold_stacks(run_dir: Path, manifest: Path) -> dict[Path, str]:
    """The test stacks of a CV run (from its ``run.json`` split file) and their split group."""
    from floodfusion.data.records import load_manifest

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    test = set(json.loads(Path(record["split_file"]).read_text(encoding="utf-8"))["test_chips"])
    return {Path(r.raster_path): r.basin_id for r in load_manifest(manifest) if r.chip_id in test}


def score_runs(run_dirs: Sequence[Path], manifest: Path, *,
               predictor: Callable[[Path], Callable[[Path], np.ndarray]]) -> dict[str, Any]:
    """Histograms of every test stack of every run; ``predictor(run_dir)`` returns ``stack -> probability``."""
    from floodfusion.config import ExperimentConfig
    from floodfusion.data.chipping import read_event_stack

    cfg = ExperimentConfig()
    runs = {}
    for run_dir in run_dirs:
        ev = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
        predict = predictor(run_dir)
        stacks = {}
        for stack, group in sorted(fold_stacks(run_dir, manifest).items()):
            raw, _, _ = read_event_stack(stack)
            scored, label, hand_m = scene_layers(raw, cfg.grid, cfg.norm)
            stacks[stack.parent.name] = {"group": group, **stack_entry(predict(stack), scored, label, hand_m)}
        runs[str(run_dir)] = {"tau": float(ev["tau_star"]), "pi0": float(ev["calibration"]["pi_deploy"]),
                              "stacks": stacks}
        LOGGER.info("%s: %d test stacks", run_dir, len(stacks))
    return {"hand_max_m": HAND_MAX_M, "runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the inference add-ons on whole CV test stacks (GPU).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="predict every test stack of the runs; write histograms")
    sc.add_argument("--runs", type=Path, nargs="+", required=True, help="CV run directories (one per fold)")
    sc.add_argument("--manifest", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--device", default=None)
    rp = sub.add_parser("report", help="the adoption decision from scored histograms")
    rp.add_argument("--scores", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "score":
        from floodfusion.case_study import _learned
        from floodfusion.train import resolve_device

        device = resolve_device(args.device)
        scores = score_runs(args.runs, args.manifest, predictor=lambda run: _learned(run, device).predict)
        with gzip.open(args.out, "wt", encoding="utf-8") as f:
            json.dump(scores, f)
        return 0
    with gzip.open(args.scores, "rt", encoding="utf-8") as f:
        result = adoption(json.load(f))
    args.out.write_text(json.dumps(result, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    LOGGER.info("macro IoU %s; adopted: %s", {k: round(v, 4) for k, v in result["macro_iou"].items()},
                [arm for arm in ARMS[1:] if result[arm]["adopted"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
