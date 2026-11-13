r"""v1.3's adaptation ladder: does adapting v1.2's model to Copernicus EMS's own labels help, and how?

The rules are ``reports/protocol_v1_3.md`` §3, committed before any rung was computed. On the seen CEMS 2023–26
benchmark's primary stratum (products drawn on radar satellites; EMSR664 left out), each activation is scored by the
fold that held out its 100 km group (``data/pool_v13``):

- **R1, zero-shot:** row 21 at its own τ*.
- **R2, recalibration:** row 21's scores, cut at the threshold minimising ``r·FN + FP`` on the pooled pixels of every
  other fold (:func:`floodfusion.crossfit.crossfit_cost_threshold`, r = 3, the cost ratio that defines τ*).
- **R3, fine-tuning:** row 21 trained further on each fold's training groups (``crossval run --init-from-root``);
  each fold model scores the pairs it held out with the v1.2 scorer (:func:`score_r3`), at its own τ*.

Every rung is read from the same binned counts (:mod:`floodfusion.histograms`), seed 42, and compared by
:func:`adopt`: a rung is adopted over the one below it only with a macro gain of at least :data:`MIN_GAIN` **and**
an improvement on more than half of the activations.

Run::

    uv run --frozen python -m floodfusion.ladder r2         # rungs 1 and 2, CPU, from the committed v1.2 scores
    uv run --frozen python -m floodfusion.ladder r3-score   # rung 3's fold models on their held-out pairs (GPU)
    uv run --frozen python -m floodfusion.ladder report     # the three rungs and the verdict (CPU)

Torch-free.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["COST_RATIO", "EXCLUDE", "MIN_GAIN", "MODEL", "PRIMARY_STRATUM", "R3_KEY", "RUN_NAME", "SEED", "adopt",
           "cluster_ious", "fold_of_activation", "fold_pairs", "ladder_report", "load_scores", "main", "render_ladder",
           "rung1", "rung2", "score_r3", "unit_histograms"]

LOGGER = logging.getLogger(__name__)

PRIMARY_STRATUM = "sar"
EXCLUDE = ("EMSR664",)
#: The miss-to-false-alarm cost ratio behind τ* = 1/(1 + r) (``configs/base.yaml``, ``eval.cost_ratio``).
COST_RATIO = 3.0
#: Row 21 in the v1.2 scores, and the seed every rung uses.
MODEL, SEED = "ours", "seed42"
MIN_GAIN = 0.005
#: Rung 3's runs: ``<runs root>/kfold/<fold>/<RUN_NAME>/<SEED>``, scored under the key :data:`R3_KEY`.
RUN_NAME, R3_KEY = "row21_fusion_context_seasonal", "r3"
_ACTIVATION = re.compile(r"(EMSR[0-9]{3,4})_", re.ASCII)
_SPEC = {"exclude": EXCLUDE, "primary_stratum": PRIMARY_STRATUM}

Hist = Mapping[str, np.ndarray]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_scores(path: Path) -> dict[str, Any]:
    from floodfusion.cems_scores_v12 import load_scores as _load

    return _load(Path(path))


def _histograms(scores: Mapping[str, Any], unit: str, model: str, seed: str,
                stratum: str = PRIMARY_STRATUM) -> dict[str, dict[str, np.ndarray]]:
    from floodfusion.cems_scores_v12 import _seed_hists, _unit_set, _units

    groups = _units(scores, {**_SPEC, "primary_stratum": stratum}, unit)
    out = {}
    for name in _unit_set(groups, gfm_covered=False):
        hist = _seed_hists(groups[name], model)[seed]
        out[name] = {"pos": hist["pos"], "neg": hist["neg"]}
    return out


def unit_histograms(scores: Mapping[str, Any], *, model: str = MODEL, seed: str = SEED,
                    stratum: str = PRIMARY_STRATUM) -> dict[str, dict[str, Any]]:
    """``{activation: {"pos", "neg"}}``: a stratum's activations with reference flood (EMSR664 out), pairs pooled."""
    return _histograms(scores, "activation", model, seed, stratum)


def fold_of_activation(chip_dir: Path, *, protocol: str = "kfold") -> dict[str, str]:
    """``{activation: fold}``: the fold whose frozen test groups hold the activation's chips."""
    from floodfusion.data.records import load_manifest

    group_of: dict[str, str] = {}
    for record in load_manifest(Path(chip_dir) / "manifest.parquet"):
        match = _ACTIVATION.match(record.event_id)
        if match is None:
            continue
        if group_of.setdefault(match.group(1), record.basin_id) != record.basin_id:
            raise ValueError(f"{match.group(1)} has chips in two groups")
    fold_of_group = {}
    for fold in json.loads((Path(chip_dir) / "cv" / "folds.json").read_text(encoding="utf-8")):
        if fold["protocol"] != protocol:
            continue
        split = json.loads((Path(chip_dir) / fold["split_file"]).read_text(encoding="utf-8"))
        fold_of_group.update({g: fold["name"] for g in split["test_basins"]})
    return {a: fold_of_group[g] for a, g in sorted(group_of.items()) if g in fold_of_group}


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------
def _counts(hist: Hist, tau: float) -> tuple[int, int, int]:
    from floodfusion import histograms

    tp, fp, fn, _ = histograms.confusion_above(hist["pos"], hist["neg"], tau)
    return tp, fp, fn


def _iou(hist: Hist, tau: float) -> float:
    tp, fp, fn = _counts(hist, tau)
    return tp / (tp + fp + fn) if tp + fp + fn else math.nan


def cluster_ious(units: Mapping[str, Hist], cut_of: Mapping[str, float],
                 cluster_of: Mapping[str, str]) -> dict[str, float]:
    """Per cluster, the IoU of its activations' pooled counts, each activation cut at its own threshold.

    A cluster (v1.1's: 100 km and 60 days, from activation extents) can span folds, so no single fold's threshold may
    be applied to all of it (``reports/v1_3_deviations.md``). An activation outside ``cluster_of`` is its own cluster.
    """
    pooled: dict[str, list[int]] = {}
    for name, hist in sorted(units.items()):
        acc = pooled.setdefault(cluster_of.get(name, name), [0, 0, 0])
        for i, n in enumerate(_counts(hist, cut_of[name])):
            acc[i] += n
    return {c: tp / (tp + fp + fn) if tp + fp + fn else math.nan for c, (tp, fp, fn) in sorted(pooled.items())}


def rung1(units: Mapping[str, Hist], *, tau: float) -> dict[str, float]:
    """Each unit's IoU at the model's own threshold."""
    return {name: _iou(h, tau) for name, h in sorted(units.items())}


def _edge_tau(k: int) -> float:
    from floodfusion import histograms

    return 1.0 / (1.0 + math.exp(-float(histograms.EDGES[k])))


def _pooled(units: Sequence[Hist]) -> dict[str, np.ndarray]:
    return {part: sum((np.asarray(u[part], dtype=np.int64) for u in units), np.int64(0)) for part in ("pos", "neg")}


def _cost_cut(units_by_fold: Mapping[str, Sequence[Hist]], cost_ratio: float) -> dict[str, float]:
    """Per fold, the bin edge minimising ``r·FN + FP`` on the other folds' pooled pixels, as a probability."""
    from floodfusion import histograms
    from floodfusion.crossfit import crossfit_cost_threshold
    from floodfusion.metrics import ConfusionMatrix

    grid = list(range(len(histograms.EDGES)))     # bin-edge indices: exact keys, no rounding of probabilities
    curves = {}
    for fold, units in units_by_fold.items():
        pooled = _pooled(units)
        curves[fold] = (grid, [ConfusionMatrix(*histograms.confusion_above(pooled["pos"], pooled["neg"], _edge_tau(k)))
                               for k in grid])
    fits = crossfit_cost_threshold(curves, cost_ratio)
    return {fold: _edge_tau(int(fit["threshold"])) for fold, fit in fits.items()}


def rung2(units: Mapping[str, Hist], fold_of: Mapping[str, str], *,
          cost_ratio: float = COST_RATIO) -> tuple[dict[str, float], dict[str, float]]:
    """Each unit's IoU at its fold's cross-fitted cost-optimal threshold; and those thresholds."""
    missing = sorted(set(units) - set(fold_of))
    if missing:
        raise KeyError(f"units in no fold: {missing}")
    by_fold: dict[str, list[Hist]] = {}
    for name, hist in units.items():
        by_fold.setdefault(fold_of[name], []).append(hist)
    cuts = _cost_cut(by_fold, cost_ratio)
    return {name: _iou(h, cuts[fold_of[name]]) for name, h in sorted(units.items())}, dict(sorted(cuts.items()))


def all_units_cut(units: Mapping[str, Hist], *, cost_ratio: float = COST_RATIO) -> float:
    """The cost-optimal threshold on every unit's pooled pixels: R2's rule for a final model (protocol v1.3 §3)."""
    from floodfusion import histograms

    pooled = _pooled(list(units.values()))
    cost = [cost_ratio * fn + fp for tp, fp, fn, _ in
            (histograms.confusion_above(pooled["pos"], pooled["neg"], _edge_tau(k))
             for k in range(len(histograms.EDGES)))]
    return _edge_tau(int(np.argmin(cost)))


def adopt(lower: Mapping[str, float], upper: Mapping[str, float], *, min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """The adoption rule: ``upper`` beats ``lower`` by ≥ ``min_gain`` macro IoU and on more than half the units."""
    if set(lower) != set(upper):
        raise ValueError("the two rungs must score the same activations")
    names = sorted(lower)
    macro_lower = float(np.mean([lower[n] for n in names]))
    macro_upper = float(np.mean([upper[n] for n in names]))
    better = sum(upper[n] > lower[n] for n in names)
    gain = macro_upper - macro_lower
    return {"n": len(names), "macro_lower": macro_lower, "macro_upper": macro_upper, "gain": gain, "better": better,
            "adopted": bool(gain >= min_gain - 1e-12 and better > len(names) / 2)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def r2_report(scores: Mapping[str, Any], fold_of: Mapping[str, str]) -> dict[str, Any]:
    """R1 and R2 per activation (the rule) and per cluster (descriptive); three seeds descriptive; the verdict."""
    from floodfusion.cems_scores import activation_clusters

    tau = 1 / (1 + COST_RATIO)
    units = unit_histograms(scores)
    r1, (r2, cuts) = rung1(units, tau=tau), rung2(units, fold_of)
    cluster_of = activation_clusters(scores["activations"])
    c1 = cluster_ious(units, dict.fromkeys(units, tau), cluster_of)
    c2 = cluster_ious(units, {a: cuts[fold_of[a]] for a in units}, cluster_of)
    three: dict[str, dict[str, float]] = {}
    for seed in ("seed42", "seed43", "seed44"):
        u = unit_histograms(scores, seed=seed)
        three[seed] = {"r1": float(np.mean(list(rung1(u, tau=1 / (1 + COST_RATIO)).values()))),
                       "r2": float(np.mean(list(rung2(u, fold_of)[0].values()))),
                       "final_cut": all_units_cut(u)}
    return {"rule": {"model": MODEL, "seed": SEED, "cost_ratio": COST_RATIO, "min_gain": MIN_GAIN,
                     "primary_stratum": PRIMARY_STRATUM, "exclude": list(EXCLUDE)},
            "fold_of": dict(sorted(fold_of.items())), "r2_cuts": cuts, "r1": r1, "r2": r2,
            "r2_over_r1": adopt(r1, r2), "clusters": {"r1": c1, "r2": c2, "r2_over_r1": adopt(c1, c2)},
            "three_seeds": three}


def render(report: Mapping[str, Any]) -> str:
    v, c = report["r2_over_r1"], report["clusters"]["r2_over_r1"]
    lines = [
        "# v1.3 ladder: rungs 1 and 2 (rung 3 pending)", "",
        "Protocol v1.3 §3, committed before any rung was computed. Seen CEMS 2023–26 benchmark, primary stratum "
        "(products drawn on radar), EMSR664 excluded; row 21, seed 42; every activation scored with its fold's "
        "threshold (fitted on the other folds). Binned counts. Exploratory: this benchmark has been seen.", "",
        f"| Rung | Per activation (K = {v['n']}) | Per cluster (K = {c['n']}), descriptive |",
        "| --- | --- | --- |",
        f"| R1, zero-shot at τ* | {v['macro_lower']:.3f} | {c['macro_lower']:.3f} |",
        f"| R2, cross-fitted cost-optimal threshold | {v['macro_upper']:.3f} | {c['macro_upper']:.3f} |", "",
        f"**R2 over R1:** gain {v['gain']:+.3f}, better in {v['better']} of {v['n']} activations: "
        f"**{'adopted' if v['adopted'] else 'not adopted'}** (rule: ≥ +{MIN_GAIN} and more than half).", "",
        "Fold thresholds (probability): " + ", ".join(f"{k} {t:.3f}" for k, t in report["r2_cuts"].items()) + ".", "",
        "Three seeds (descriptive): " + "; ".join(f"{s} R1 {d['r1']:.3f}, R2 {d['r2']:.3f}"
                                                  for s, d in report["three_seeds"].items()) + ".", "",
        "| Activation | Fold | R1 | R2 |", "| --- | --- | --- | --- |",
        *(f"| {a} | {report['fold_of'][a]} | {report['r1'][a]:.3f} | {report['r2'][a]:.3f} |" for a in report["r1"]),
        "", "*Regenerate:* `uv run --frozen python -m floodfusion.ladder r2`.", "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rung 3
# ---------------------------------------------------------------------------
def fold_pairs(case_root: Path, fold_of: Mapping[str, str]) -> dict[str, list[str]]:
    """``{fold: [pair key, ...]}``: every pair, of every stratum, of the activations each fold holds out."""
    out: dict[str, list[str]] = {}
    for pair_file in sorted(Path(case_root).glob("EMSR*_AOI*/pair.json")):
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
        fold = fold_of.get(str(pair["activation"]))
        if fold is not None:
            out.setdefault(fold, []).append(str(pair["key"]))
    return dict(sorted(out.items()))


def score_r3(runs_root: Path, case_root: Path, fold_of: Mapping[str, str], *, device: Any = None,
             records: Path | None = None) -> dict[str, Any]:
    """Each fold's fine-tuned model scores only the pairs that fold held out; one scores document, keyed ``r3``."""
    from floodfusion import cems_scores_v12
    from floodfusion.cems_scores import ModelSpec

    merged: dict[str, Any] = {"pairs": {}, "activations": {}, "priors": {}, "fold_priors": {}, "models": {R3_KEY: {
        "name": "R3: row 21 fine-tuned on each fold's training groups (protocol v1.3 §3)", "runs": [],
        "kind": "learned", "role": "test"}}}
    for fold, keys in fold_pairs(case_root, fold_of).items():
        run = Path(runs_root) / "kfold" / fold / RUN_NAME / SEED
        spec = {"models": [ModelSpec(R3_KEY, f"R3 {fold}", (str(run),))], "configs_dir": "configs", "overlays": ()}
        part = cems_scores_v12.score(spec, Path(case_root), device, records=records, pairs_only=set(keys))
        merged["pairs"].update(part["pairs"])
        merged["activations"].update(part["activations"])
        merged["fold_priors"][fold] = part["priors"][R3_KEY][SEED]
        merged["models"][R3_KEY]["runs"].append(str(run))
        LOGGER.info("R3 %s: %d pairs scored", fold, len(part["pairs"]))
    return merged


def _check_same_stacks(v12: Mapping[str, Any], r3: Mapping[str, Any]) -> None:
    for key, entry in r3["pairs"].items():
        mine = v12["pairs"].get(key)
        if mine is None or any(mine[f] != entry[f] for f in ("scored_px", "flood_px")):
            raise ValueError(f"{key}: rung 3 was scored on different stacks than rungs 1 and 2")


def _macro(values: Mapping[str, float]) -> float | None:
    finite = [v for v in values.values() if not math.isnan(v)]
    return float(np.mean(finite)) if finite else None


def _threshold_free(units: Mapping[str, Hist]) -> dict[str, float | None]:
    from floodfusion import histograms

    return {"ap": _macro({a: histograms.average_precision(h["pos"], h["neg"]) for a, h in units.items()}),
            "oracle_iou": _macro({a: histograms.oracle_iou(h["pos"], h["neg"])[0] for a, h in units.items()})}


def ladder_report(v12: Mapping[str, Any], r3: Mapping[str, Any], fold_of: Mapping[str, str]) -> dict[str, Any]:
    """The three rungs per activation, the two adoption verdicts, v1.3's model, and the descriptive rows."""
    from floodfusion.cems_scores import activation_clusters

    _check_same_stacks(v12, r3)
    tau = 1 / (1 + COST_RATIO)
    fold_tau = {f: float(p["tau"]) for f, p in r3["fold_priors"].items()}
    u12, u3 = unit_histograms(v12), unit_histograms(r3, model=R3_KEY)
    r1, (r2, cuts) = rung1(u12, tau=tau), rung2(u12, fold_of)
    r3_ious = {a: _iou(h, fold_tau[fold_of[a]]) for a, h in sorted(u3.items())}
    v2 = adopt(r1, r2)
    base_name, base = ("r2", r2) if v2["adopted"] else ("r1", r1)
    v3 = {**adopt(base, r3_ious), "base": base_name}
    model = "R3" if v3["adopted"] else ("R2" if v2["adopted"] else None)
    cluster_of = activation_clusters(v12["activations"])
    clusters = {"r1": cluster_ious(u12, dict.fromkeys(u12, tau), cluster_of),
                "r2": cluster_ious(u12, {a: cuts[fold_of[a]] for a in u12}, cluster_of),
                "r3": cluster_ious(u3, {a: fold_tau[fold_of[a]] for a in u3}, cluster_of)}
    o12, o3 = unit_histograms(v12, stratum="optical"), unit_histograms(r3, model=R3_KEY, stratum="optical")
    optical = {"n": len(o3), "r1": _macro(rung1(o12, tau=tau)),
               "r3": _macro({a: _iou(h, fold_tau[fold_of[a]]) for a, h in o3.items()})}
    return {"rule": {"model": MODEL, "seed": SEED, "cost_ratio": COST_RATIO, "min_gain": MIN_GAIN,
                     "primary_stratum": PRIMARY_STRATUM, "exclude": list(EXCLUDE)},
            "fold_of": dict(sorted(fold_of.items())), "r2_cuts": cuts, "r3_taus": dict(sorted(fold_tau.items())),
            "r1": r1, "r2": r2, "r3": r3_ious, "verdicts": {"r2_over_r1": v2, "r3_over_base": v3},
            "v13_model": model,
            "descriptive": {"r3_over_r2": adopt(r2, r3_ious), "clusters": {
                **clusters, "r3_over_r1": adopt(clusters["r1"], clusters["r3"])},
                "threshold_free": {"r1": _threshold_free(u12), "r3": _threshold_free(u3)}, "optical": optical}}


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_ladder(report: Mapping[str, Any]) -> str:
    v2, v3 = report["verdicts"]["r2_over_r1"], report["verdicts"]["r3_over_base"]
    d = report["descriptive"]
    c, tf, opt = d["clusters"], d["threshold_free"], d["optical"]
    word = {True: "adopted", False: "not adopted"}
    model = {"R3": "R3, row 21 fine-tuned on Copernicus labels (finals on all groups follow, protocol v1.3 §3)",
             "R2": "R2, row 21 with the refitted threshold", None: "none: no rung was adopted; H8 is not run"}
    lines = [
        "# v1.3 ladder: does training on Copernicus's own labels help?", "",
        "Protocol v1.3 §3, committed before any rung was computed. Seen CEMS 2023–26 benchmark, primary stratum "
        "(products drawn on radar), EMSR664 excluded; seed 42. Every activation is scored by the fold that held out "
        "its 100 km group; binned counts. Exploratory: this benchmark has been seen. Only H8, on the blind floods, "
        "is confirmatory.", "",
        f"| Rung | Macro IoU, {v2['n']} activations | Per cluster (descriptive) |", "| --- | --- | --- |",
        f"| R1, zero-shot at τ* | {v2['macro_lower']:.3f} | {_macro(c['r1']):.3f} |",
        f"| R2, cross-fitted cost-optimal threshold | {v2['macro_upper']:.3f} | {_macro(c['r2']):.3f} |",
        f"| R3, fine-tuned on each fold's training groups | {v3['macro_upper']:.3f} | {_macro(c['r3']):.3f} |", "",
        f"- **R2 over R1:** {v2['gain']:+.3f}, better in {v2['better']} of {v2['n']}: **{word[v2['adopted']]}**.",
        f"- **R3 over {v3['base'].upper()}** (the highest rung adopted below it): {v3['gain']:+.3f}, better in "
        f"{v3['better']} of {v3['n']}: **{word[v3['adopted']]}**.",
        f"- Rule: a gain of at least +{MIN_GAIN} and better on more than half of the activations.", "",
        f"**v1.3's model:** {model[report['v13_model']]}.", "",
        "Descriptive, not part of the rule:", "",
        f"- R3 over R2: {d['r3_over_r2']['gain']:+.3f}, better in {d['r3_over_r2']['better']} of "
        f"{d['r3_over_r2']['n']}.",
        f"- Per cluster, R3 over R1: {c['r3_over_r1']['gain']:+.3f}, better in {c['r3_over_r1']['better']} of "
        f"{c['r3_over_r1']['n']} clusters.",
        f"- Threshold-free: AP R1 {_f(tf['r1']['ap'])}, R3 {_f(tf['r3']['ap'])}; oracle IoU R1 "
        f"{_f(tf['r1']['oracle_iou'])}, R3 {_f(tf['r3']['oracle_iou'])}.",
        f"- Products drawn on optical images ({opt['n']} activations): R1 {_f(opt['r1'])}, R3 {_f(opt['r3'])}.",
        "- Fold thresholds: R2 " + ", ".join(f"{k} {t:.3f}" for k, t in report["r2_cuts"].items())
        + "; R3's τ* " + ", ".join(f"{k} {t:.3f}" for k, t in report["r3_taus"].items()) + ".", "",
        "| Activation | Fold | R1 | R2 | R3 |", "| --- | --- | --- | --- | --- |",
        *(f"| {a} | {report['fold_of'][a]} | {report['r1'][a]:.3f} | {report['r2'][a]:.3f} | {report['r3'][a]:.3f} |"
          for a in report["r1"]),
        "", "*Regenerate:* `uv run --frozen python -m floodfusion.ladder report`.", "",
    ]
    return "\n".join(lines)


def _write(report: Mapping[str, Any], out: Path, text: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    out.with_suffix(".md").write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v1.3's adaptation ladder (protocol v1.3 §3).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r2 = sub.add_parser("r2", help="rungs 1 and 2 from the committed v1.2 scores (CPU)")
    r3s = sub.add_parser("r3-score", help="rung 3: each fold model on the pairs it held out (GPU)")
    rep = sub.add_parser("report", help="the three rungs and the verdict (CPU)")
    for p in (r2, r3s, rep):
        p.add_argument("--chips", type=Path, default=Path("data/pool_v13/chips"))
    for p in (r2, rep):
        p.add_argument("--scores", type=Path, default=Path("reports/v12_cems_rebuilt_scores.json.gz"))
    r2.add_argument("--out", type=Path, default=Path("reports/v13_ladder_r2.json"))
    r3s.add_argument("--runs", type=Path, default=Path("runs/v13/ladder"))
    r3s.add_argument("--case-root", type=Path, default=Path("data/cems_benchmark_v12"))
    r3s.add_argument("--records", type=Path, default=Path("data/cems_records"))
    r3s.add_argument("--device", default=None)
    r3s.add_argument("--out", type=Path, default=Path("reports/v13_r3_scores.json.gz"))
    rep.add_argument("--r3", type=Path, default=Path("reports/v13_r3_scores.json.gz"))
    rep.add_argument("--out", type=Path, default=Path("reports/v13_ladder.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fold_of = fold_of_activation(args.chips)
    if args.cmd == "r2":
        report = r2_report(load_scores(args.scores), fold_of)
        _write(report, args.out, render(report))
        v = report["r2_over_r1"]
        LOGGER.info("R1 %.3f, R2 %.3f: %s", v["macro_lower"], v["macro_upper"], "adopted" if v["adopted"] else
                    "not adopted")
    elif args.cmd == "r3-score":
        from floodfusion.cems_scores_v12 import write_scores

        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_scores(score_r3(args.runs, args.case_root, fold_of, device=args.device, records=args.records), args.out)
    else:
        report = ladder_report(load_scores(args.scores), load_scores(args.r3), fold_of)
        _write(report, args.out, render_ladder(report))
        LOGGER.info("v1.3's model: %s", report["v13_model"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
