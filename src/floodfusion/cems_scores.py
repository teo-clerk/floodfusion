r"""Score models on the CEMS benchmark and report the pre-registered tests.

``score``
    Every model of ``configs/datasets/cems_models.yaml`` predicts every ingested tile of every eligible
    pair (:mod:`floodfusion.cems_benchmark`); confusion counts at the model's own τ* (zero-shot:
    the calibration comes from its training dataset's validation set) and on an oracle
    threshold grid are summed per pair, seeds pooled. Scored pixels: inside the product's
    observed region, valid SAR, outside permanent water. Each pair also records the share of
    its scored pixels with clear optical.

``report``
    From the scores alone. The **activation** is the unit; its AOIs pool. The confirmatory
    family is the spec's ``tests`` — exactly those, Holm-adjusted together, on the primary
    stratum (products drawn on radar satellites; activations named in ``exclude`` left out).
    A test whose subset has fewer than :data:`MIN_UNITS_FOR_VERDICT` activations gets no
    verdict. A non-inferiority test passes when the lower bound of the paired 95 % interval
    lies above ``-margin``. Everything else is descriptive.

Products: © European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import ConfusionMatrix

__all__ = ["MIN_UNITS_FOR_VERDICT", "GapTest", "ModelSpec", "load_spec", "main", "report", "run_tests"]

LOGGER = logging.getLogger(__name__)

#: As everywhere in this project: no significance verdict below 8 units (``reporting``).
MIN_UNITS_FOR_VERDICT = 8
#: Thresholds of the oracle grid (as the case study).
_GRID = tuple(round(i / 20, 2) for i in range(1, 20))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    runs: tuple[str, ...]          # run directories, one per seed (learned) or one (baseline)
    kind: str = "learned"          # learned | baseline


@dataclass(frozen=True)
class GapTest:
    id: str
    question: str
    a: str
    b: str                          # the gap is b - a
    subset: str = "all"             # all | optical | sar_only
    margin: float | None = None     # non-inferiority margin (b - a > -margin), else two-sided


def load_spec(path: Path | str) -> dict[str, Any]:
    """The pre-registered models, tests and strata (``configs/datasets/cems_models.yaml``)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = [ModelSpec(m["key"], m["name"], tuple(m["runs"]), m.get("kind", "learned")) for m in raw["models"]]
    tests = [GapTest(t["id"], t["question"], t["a"], t["b"], t.get("subset", "all"), t.get("margin"))
             for t in raw["tests"]]
    keys = {m.key for m in models}
    unknown = sorted({k for t in tests for k in (t.a, t.b)} - keys)
    if unknown:
        raise ValueError(f"tests name unknown models {unknown}")
    return {"models": models, "tests": tests, "primary_stratum": raw.get("primary_stratum", "sar"),
            "exclude": tuple(raw.get("exclude", ())), "optical_usable": float(raw.get("optical_usable", 0.5)),
            "sar_only_below": float(raw.get("sar_only_below", 0.1)), "configs_dir": raw.get("configs_dir", "configs"),
            "overlays": tuple(raw.get("overlays", ()))}


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def _models(spec: Mapping[str, Any], device: Any) -> list[tuple[str, Any]]:
    """``(key, case_study model)`` for every seed of every model (seeds share a key and pool)."""
    from floodfusion.case_study import _baseline, _learned

    out = []
    for m in spec["models"]:
        for run in m.runs:
            if m.kind == "baseline":
                model = _baseline(Path(run), Path(spec["configs_dir"]), [Path(o) for o in spec["overlays"]])
            else:
                model = _learned(Path(run), device)
            out.append((m.key, replace(model, name=m.key)))
    return out


def _tile_masks(stack: Path, pair: Mapping[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    from floodfusion.config import ValidityFlag
    from floodfusion.data.chipping import CLEAR_QUALITY, QUALITY_SCALE, read_event_stack
    from floodfusion.data.encoding import ChipBand
    from floodfusion.data.ingest import rasterise_vector

    raw, grid, _ = read_event_stack(stack)
    valid = (raw[ChipBand.VALIDITY].astype(np.int64) & ~int(ValidityFlag.OUTSIDE_AOI)) == 0
    scored = valid & rasterise_vector(Path(pair["region"]), grid)
    label = rasterise_vector(Path(pair["label"]), grid)
    has_optical = bool(json.loads(stack.with_name("event.json").read_text(encoding="utf-8")).get("has_optical"))
    clear = (raw[ChipBand.OPTICAL_QUALITY] >= int(round(CLEAR_QUALITY * QUALITY_SCALE))) if has_optical \
        else np.zeros(scored.shape, dtype=bool)
    return grid, scored, label, clear


def _cms(prob: np.ndarray, tau: float, calibrated: bool, label: np.ndarray, scored: np.ndarray) -> list[list[int]]:
    from floodfusion.case_study import _score

    out = []
    for t in (tau, *(_GRID if calibrated else ())):
        cm = _score(np.nan_to_num(prob, nan=0.0) > t, label, scored)
        out.append([cm.tp, cm.fp, cm.fn, cm.tn])
    return out


def score(spec: Mapping[str, Any], case_root: Path, device: Any) -> dict[str, Any]:
    """Per pair: summed confusion counts per model key, scored pixels, clear-optical share."""
    from floodfusion.train import resolve_device

    device = resolve_device(device)
    models = _models(spec, device)
    pairs: dict[str, Any] = {}
    for pair_file in sorted(case_root.glob("EMSR*_AOI*/pair.json")):
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {"pair": {k: pair[k] for k in ("activation", "aoi", "product", "stratum", "sensors",
                                                               "offset_h", "pass_time", "product_time")},
                                 "scored_px": 0, "clear_px": 0, "flood_px": 0, "models": {}}
        for stack in sorted((pair_file.parent / "events").glob("*/stack.tif")):
            if not stack.with_name("event.json").is_file():
                continue
            _, scored, label, clear = _tile_masks(stack, pair)
            if not scored.any():
                continue
            entry["scored_px"] += int(scored.sum())
            entry["clear_px"] += int((clear & scored).sum())
            entry["flood_px"] += int((label & scored).sum())
            for key, model in models:
                cms = _cms(model.predict(stack), model.tau, model.calibrated, label, scored)
                store = entry["models"].setdefault(key, None)
                entry["models"][key] = cms if store is None else [[x + y for x, y in zip(a, b)] for a, b in zip(store, cms)]
        pairs[pair["key"]] = entry
        LOGGER.info("%s: %d scored px, %.0f %% clear optical", pair["key"], entry["scored_px"],
                    100 * entry["clear_px"] / max(entry["scored_px"], 1))
    return {"pairs": pairs, "models": {m.key: {"name": m.name, "runs": list(m.runs), "kind": m.kind}
                                       for m in spec["models"]}}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _cm(counts: Sequence[int]) -> ConfusionMatrix:
    return ConfusionMatrix(*(int(c) for c in counts))


def activations(scores: Mapping[str, Any], spec: Mapping[str, Any], *, stratum: str | None) -> dict[str, dict[str, Any]]:
    """Pairs pooled per activation (optionally one stratum; ``exclude`` dropped)."""
    out: dict[str, dict[str, Any]] = {}
    for entry in scores["pairs"].values():
        pair = entry["pair"]
        if pair["activation"] in spec["exclude"] or (stratum is not None and pair["stratum"] != stratum):
            continue
        if not entry["scored_px"]:
            continue
        unit = out.setdefault(pair["activation"], {"scored_px": 0, "clear_px": 0, "flood_px": 0, "models": {}})
        for k in ("scored_px", "clear_px", "flood_px"):
            unit[k] += entry[k]
        for key, cms in entry["models"].items():
            have = unit["models"].get(key)
            unit["models"][key] = cms if have is None else [[x + y for x, y in zip(a, b)] for a, b in zip(have, cms)]
    return out


def _subset(units: Mapping[str, Mapping[str, Any]], subset: str, spec: Mapping[str, Any]) -> list[str]:
    def share(u: Mapping[str, Any]) -> float:
        return u["clear_px"] / max(u["scored_px"], 1)

    if subset == "optical":
        return sorted(a for a, u in units.items() if share(u) >= spec["optical_usable"])
    if subset == "sar_only":
        return sorted(a for a, u in units.items() if share(u) < spec["sar_only_below"])
    return sorted(units)


def run_tests(units: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any], *, seed: int = 0) -> list[dict[str, Any]]:
    """The confirmatory family: per-activation IoU gaps, sign-flip p, Holm, paired bootstrap CI."""
    from floodfusion.metrics import holm_adjust, paired_bootstrap_delta, sign_flip_p_value

    results = []
    for t in spec["tests"]:
        keys = [a for a in _subset(units, t.subset, spec) if t.a in units[a]["models"] and t.b in units[a]["models"]]
        per_a = {k: _cm(units[k]["models"][t.a][0]) for k in keys}
        per_b = {k: _cm(units[k]["models"][t.b][0]) for k in keys}
        deltas = [per_b[k].iou - per_a[k].iou for k in keys]
        row: dict[str, Any] = {"id": t.id, "question": t.question, "a": t.a, "b": t.b, "subset": t.subset,
                               "margin": t.margin, "units": keys, "k": len(keys),
                               "wins": int(sum(d > 0 for d in deltas if math.isfinite(d)))}
        if keys:
            delta, lo, hi = paired_bootstrap_delta(per_a, per_b, statistic="macro", seed=seed)
            row.update({"delta": delta, "ci": [lo, hi], "p": sign_flip_p_value(deltas)})
        else:
            row.update({"delta": math.nan, "ci": [math.nan, math.nan], "p": math.nan})
        results.append(row)
    eligible = {r["id"]: r["p"] for r in results if r["k"] >= MIN_UNITS_FOR_VERDICT and r["margin"] is None}
    holm = holm_adjust(eligible)
    for r in results:
        if r["k"] < MIN_UNITS_FOR_VERDICT:
            r["verdict"] = f"none: K = {r['k']} < {MIN_UNITS_FOR_VERDICT}"
        elif r["margin"] is not None:
            ok = math.isfinite(r["ci"][0]) and r["ci"][0] > -r["margin"]
            r["verdict"] = f"non-inferior (95 % lower bound {r['ci'][0]:+.3f} > -{r['margin']})" if ok else \
                f"not shown non-inferior (95 % lower bound {r['ci'][0]:+.3f} <= -{r['margin']})"
        else:
            r["holm_p"] = holm[r["id"]]
            r["verdict"] = ("significant" if holm[r["id"]] < 0.05 else "not significant") + f" (Holm p = {holm[r['id']]:.3f})"
    return results


def _fmt(x: float) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.3f}"


def report(scores: Mapping[str, Any], spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Write ``out`` (Markdown) and ``out.with_suffix('.json')`` from the scores alone."""
    from floodfusion.metrics import basin_bootstrap_ci, macro_iou
    from floodfusion.runs import write_json

    primary = activations(scores, spec, stratum=spec["primary_stratum"])
    tests = run_tests(primary, spec)
    names = scores["models"]
    lines = [
        "# CEMS benchmark — independent flood maps, selected by a rule fixed in advance", "",
        "Zero-shot: no model saw any of these events; each uses the τ* and calibration of its own training data. "
        "The **activation** is the unit (its AOIs pool); seeds pool per activation. Primary stratum: products "
        f"drawn on **{spec['primary_stratum']}** satellites; excluded: {', '.join(spec['exclude']) or 'none'}. "
        "Protocol: `reports/protocol_v1_1.md`. Products: © European Union, Copernicus Emergency Management Service.", "",
        "## Pre-registered tests", "",
        "| Test | Question | B − A | Subset | K | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in tests:
        lines.append(f"| {r['id']} | {r['question']} | {names[r['b']]['name']} − {names[r['a']]['name']} | {r['subset']} | "
                     f"{r['k']} | {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['wins']}/{r['k']} | "
                     f"{_fmt(r['p'])} | {r['verdict']} |")
    lines += ["", "## Every model, primary stratum (macro IoU over activations at τ* [95 % CI]; oracle threshold)", "",
              "| Model | IoU at τ* | IoU, oracle τ | K |", "| --- | --- | --- | --- |"]
    summary: dict[str, Any] = {}
    for key, meta in names.items():
        per = {a: _cm(u["models"][key][0]) for a, u in primary.items() if key in u["models"]}
        if not per:
            continue
        ci = basin_bootstrap_ci(per, statistic="macro") if len(per) > 1 else (math.nan, math.nan, math.nan)
        oracle = _oracle(primary, key)
        summary[key] = {"iou": macro_iou(per), "ci": list(ci[1:]) if len(ci) == 3 else ci, "oracle": oracle, "k": len(per)}
        lines.append(f"| {meta['name']} | {_fmt(macro_iou(per))} [{_fmt(summary[key]['ci'][0])}, "
                     f"{_fmt(summary[key]['ci'][1])}] | {_fmt(oracle)} | {len(per)} |")
    lines += ["", "## Activations", "", "| Activation | Stratum | Scored km² | Flood prevalence | Clear optical |",
              "| --- | --- | --- | --- | --- |"]
    everything = activations(scores, {**spec, "exclude": ()}, stratum=None)
    strata = {e["pair"]["activation"]: e["pair"]["stratum"] for e in scores["pairs"].values()}
    for a, u in sorted(everything.items()):
        lines.append(f"| {a} | {strata[a]} | {u['scored_px'] / 1e4:.0f} | {u['flood_px'] / max(u['scored_px'], 1):.3f} | "
                     f"{u['clear_px'] / max(u['scored_px'], 1):.2f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"tests": tests, "models": summary, "units": sorted(primary)}
    write_json(out.with_suffix(".json"), result)
    return result


def _oracle(units: Mapping[str, Mapping[str, Any]], key: str) -> float:
    """Macro IoU over activations at the best single grid threshold (not deployable)."""
    best = math.nan
    n = max((len(u["models"][key]) for u in units.values() if key in u["models"]), default=0)
    for i in range(1, n):
        ious = [_cm(u["models"][key][i]).iou for u in units.values() if key in u["models"]]
        ious = [x for x in ious if math.isfinite(x)]
        if ious:
            value = float(np.mean(ious))
            best = value if not math.isfinite(best) else max(best, value)
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score models on the CEMS benchmark; report the pre-registered tests.")
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models.yaml"))
    sc.add_argument("--case-root", type=Path, default=Path("data/cems_benchmark"))
    sc.add_argument("--out", type=Path, default=Path("reports/cems_benchmark_scores.json"))
    sc.add_argument("--device", default=None)
    rp = sub.add_parser("report")
    rp.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models.yaml"))
    rp.add_argument("--scores", type=Path, default=Path("reports/cems_benchmark_scores.json"))
    rp.add_argument("--out", type=Path, default=Path("reports/cems_benchmark.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = load_spec(args.spec)
    if args.command == "score":
        from floodfusion.runs import write_json
        from floodfusion.utils.env import git_state

        write_json(args.out, {**score(spec, args.case_root, args.device), "git": git_state()})
        return 0
    report(json.loads(args.scores.read_text(encoding="utf-8")), spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
