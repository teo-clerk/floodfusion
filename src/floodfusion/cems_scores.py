r"""Score models on the CEMS benchmark and report the pre-registered tests.

``score``
    Every model of ``configs/datasets/cems_models.yaml`` predicts every ingested tile of every
    eligible pair (:mod:`floodfusion.cems_benchmark`). Zero-shot: each model uses the τ* and the
    calibration of its own training data — for a model with an optical input, the **SAR-only**
    calibration wherever a window had no usable optical (:func:`floodfusion.case_study._learned`).
    Confusion counts at τ* and on an oracle threshold grid are kept **per seed** and per pair. Scored
    pixels: inside the product's observed region, valid SAR, outside permanent water. Each pair also
    records the share of its scored pixels with clear optical.

``report``
    From the scores alone; seeds pool here. Two units, both pre-registered
    (``reports/protocol_v1_1.md``, amendment 1): the **activation** (its AOIs pool) and the
    **cluster** — activations within :data:`CLUSTER_KM` and :data:`CLUSTER_DAYS` of each other
    (single linkage), which may be one weather system. The confirmatory family is the spec's
    two-sided tests that reach :data:`MIN_UNITS_FOR_VERDICT` units, Holm-adjusted within each unit
    definition; a hypothesis is **supported** only if its test holds at both. Non-inferiority
    tests (P2) are exact one-sided sign-flip tests of :math:`d + m > 0` at :data:`NI_ALPHA`,
    outside the Holm family. Everything else — subsets of partial cloud, sensitivity analyses,
    the per-model table — is descriptive.

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

__all__ = ["CLUSTER_DAYS", "CLUSTER_KM", "MIN_UNITS_FOR_VERDICT", "NI_ALPHA", "GapTest", "ModelSpec",
           "activation_clusters", "load_spec", "main", "one_sided_sign_flip_p", "report", "run_tests", "units"]

LOGGER = logging.getLogger(__name__)

#: As everywhere in this project: no significance verdict below 8 units (``reporting``).
MIN_UNITS_FOR_VERDICT = 8
#: Activations this close in space and time are one cluster (the Kuro Siwo grouping rule, in time too).
CLUSTER_KM = 100.0
CLUSTER_DAYS = 60
#: One-sided level of the non-inferiority test.
NI_ALPHA = 0.025
#: Sensitivity: units with less flood than this (km²) in their scored region are noise-dominated.
MIN_FLOOD_KM2 = 1.0
#: Thresholds of the oracle grid (as the case study).
_GRID = tuple(round(i / 20, 2) for i in range(1, 20))
_PX_KM2 = 1e-4                    # one 10 m pixel
_MC_PATTERNS = 200_000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    runs: tuple[str, ...]          # run directories, one per seed (learned) or one (baseline)
    kind: str = "learned"          # learned | baseline
    role: str = "test"             # test (named by a test) | descriptive


@dataclass(frozen=True)
class GapTest:
    id: str
    question: str
    a: str
    b: str                          # the gap is b - a
    subset: str = "all"             # all | optical | sar_only | partial
    margin: float | None = None     # non-inferiority margin (b - a > -margin), else two-sided


def load_spec(path: Path | str) -> dict[str, Any]:
    """The pre-registered models, tests, strata and sensitivity analyses (``configs/datasets/cems_models.yaml``)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = [ModelSpec(m["key"], m["name"], tuple(m["runs"]), m.get("kind", "learned"), m.get("role", "test"))
              for m in raw["models"]]
    tests = [GapTest(t["id"], t["question"], t["a"], t["b"], t.get("subset", "all"), t.get("margin"))
             for t in raw["tests"]]
    keys = {m.key for m in models}
    unknown = sorted({k for t in tests for k in (t.a, t.b)} - keys)
    if unknown:
        raise ValueError(f"tests name unknown models {unknown}")
    return {"models": models, "tests": tests, "primary_stratum": raw.get("primary_stratum", "sar"),
            "exclude": tuple(raw.get("exclude", ())), "optical_usable": float(raw.get("optical_usable", 0.5)),
            "sar_only_below": float(raw.get("sar_only_below", 0.1)), "configs_dir": raw.get("configs_dir", "configs"),
            "overlays": tuple(raw.get("overlays", ())),
            "overlapping_pairs": tuple(raw.get("overlapping_pairs", ()))}


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def _models(spec: Mapping[str, Any], device: Any) -> list[tuple[str, str, Any]]:
    """``(key, run, case_study model)`` for every seed of every model."""
    from floodfusion.case_study import _baseline, _learned

    out = []
    for m in spec["models"]:
        for run in m.runs:
            if m.kind == "baseline":
                model = _baseline(Path(run), Path(spec["configs_dir"]), [Path(o) for o in spec["overlays"]])
            else:
                model = _learned(Path(run), device)
            out.append((m.key, Path(run).name, replace(model, name=m.key)))
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


def _add(a: list[list[int]] | None, b: list[list[int]]) -> list[list[int]]:
    return b if a is None else [[x + y for x, y in zip(u, v)] for u, v in zip(a, b)]


def score(spec: Mapping[str, Any], case_root: Path, device: Any, *, records: Path | None = None) -> dict[str, Any]:
    """Per pair: confusion counts per model key and seed, scored pixels, clear-optical share."""
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
            for key, run, model in models:
                per_seed = entry["models"].setdefault(key, {})
                per_seed[run] = _add(per_seed.get(run), _cms(model.predict(stack), model.tau, model.calibrated,
                                                            label, scored))
        pairs[pair["key"]] = entry
        LOGGER.info("%s: %d scored px, %.0f %% clear optical", pair["key"], entry["scored_px"],
                    100 * entry["clear_px"] / max(entry["scored_px"], 1))
    events = {}
    if records is not None:
        for code in sorted({e["pair"]["activation"] for e in pairs.values()}):
            rec = json.loads((records / f"{code}.json").read_text(encoding="utf-8"))
            events[code] = {"eventTime": rec["eventTime"], "extent": rec["extent"]}
    return {"pairs": pairs, "activations": events,
            "models": {m.key: {"name": m.name, "runs": list(m.runs), "kind": m.kind, "role": m.role}
                       for m in spec["models"]}}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def _cm(counts: Sequence[int]) -> ConfusionMatrix:
    return ConfusionMatrix(*(int(c) for c in counts))


def activation_clusters(events: Mapping[str, Mapping[str, Any]], *, km: float = CLUSTER_KM,
                        days: int = CLUSTER_DAYS) -> dict[str, str]:
    """``{activation: cluster}``: single linkage of activations within ``km`` and ``days`` of each other.

    A cluster is named after its alphabetically first activation.
    """
    import shapely
    from pyproj import Geod
    from shapely.ops import nearest_points

    from floodfusion.cems_benchmark import _time

    geod = Geod(ellps="WGS84")
    codes = sorted(events)
    parent = {c: c for c in codes}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    geoms = {c: shapely.from_wkt(str(events[c]["extent"])) for c in codes}
    times = {c: _time(events[c]["eventTime"]) for c in codes}
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if abs((times[a] - times[b]).total_seconds()) > days * 86400:
                continue
            if geoms[a].intersects(geoms[b]):
                close = True
            else:
                pa, pb = nearest_points(geoms[a], geoms[b])
                close = geod.inv(pa.x, pa.y, pb.x, pb.y)[2] / 1000.0 <= km
            if close:
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    return {c: find(c) for c in codes}


def units(scores: Mapping[str, Any], spec: Mapping[str, Any], *, stratum: str | None, unit: str = "activation",
          exclude_pairs: Sequence[str] = (), min_flood_km2: float = 0.0) -> dict[str, dict[str, Any]]:
    """Pairs pooled per unit (activation or cluster), seeds pooled; ``exclude`` and ``exclude_pairs`` dropped."""
    cluster_of = activation_clusters(scores["activations"]) if unit == "cluster" else {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in scores["pairs"].items():
        pair = entry["pair"]
        if pair["activation"] in spec["exclude"] or key in exclude_pairs or not entry["scored_px"]:
            continue
        if stratum is not None and pair["stratum"] != stratum:
            continue
        name = cluster_of.get(pair["activation"], pair["activation"]) if unit == "cluster" else pair["activation"]
        u = out.setdefault(name, {"scored_px": 0, "clear_px": 0, "flood_px": 0, "models": {}, "members": set()})
        u["members"].add(pair["activation"])
        for k in ("scored_px", "clear_px", "flood_px"):
            u[k] += entry[k]
        for model, per_seed in entry["models"].items():
            pooled = None
            for cms in per_seed.values():
                pooled = _add(pooled, cms)
            u["models"][model] = _add(u["models"].get(model), pooled)
    return {k: v for k, v in out.items() if v["flood_px"] * _PX_KM2 >= min_flood_km2}


def _subset(us: Mapping[str, Mapping[str, Any]], subset: str, spec: Mapping[str, Any]) -> list[str]:
    def share(u: Mapping[str, Any]) -> float:
        return u["clear_px"] / max(u["scored_px"], 1)

    if subset == "optical":
        return sorted(a for a, u in us.items() if share(u) >= spec["optical_usable"])
    if subset == "sar_only":
        return sorted(a for a, u in us.items() if share(u) < spec["sar_only_below"])
    if subset == "partial":
        return sorted(a for a, u in us.items() if spec["sar_only_below"] <= share(u) < spec["optical_usable"])
    return sorted(us)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def one_sided_sign_flip_p(values: Sequence[float], *, seed: int = 0) -> float:
    """P(mean of randomly signed ``values`` >= observed mean): exact for K <= 20, Monte Carlo above."""
    d = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    k = len(d)
    if k == 0:
        return float("nan")
    observed = d.mean()
    if k <= 20:
        signs = 1 - 2 * ((np.arange(2**k)[:, None] >> np.arange(k)) & 1)
        return float(((signs * d).mean(axis=1) >= observed - 1e-12).mean())
    signs = np.random.default_rng(seed).choice((-1, 1), size=(_MC_PATTERNS, k))
    return float((((signs * d).mean(axis=1) >= observed - 1e-12).sum() + 1) / (_MC_PATTERNS + 1))


def run_tests(us: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any], *, seed: int = 0) -> list[dict[str, Any]]:
    """Per test: per-unit IoU gaps, sign-flip p (Holm over the eligible two-sided tests), paired bootstrap CI."""
    from floodfusion.metrics import holm_adjust, paired_bootstrap_delta, sign_flip_p_value

    results = []
    for t in spec["tests"]:
        keys = []
        for k in _subset(us, t.subset, spec):
            ms = us[k]["models"]
            if t.a in ms and t.b in ms and math.isfinite(_cm(ms[t.a][0]).iou) and math.isfinite(_cm(ms[t.b][0]).iou):
                keys.append(k)
        per_a = {k: _cm(us[k]["models"][t.a][0]) for k in keys}
        per_b = {k: _cm(us[k]["models"][t.b][0]) for k in keys}
        deltas = [per_b[k].iou - per_a[k].iou for k in keys]
        row: dict[str, Any] = {"id": t.id, "question": t.question, "a": t.a, "b": t.b, "subset": t.subset,
                               "margin": t.margin, "units": keys, "k": len(keys), "wins": int(sum(d > 0 for d in deltas)),
                               "per_unit_delta": dict(zip(keys, deltas))}
        if keys:
            delta, lo, hi = paired_bootstrap_delta(per_a, per_b, statistic="macro", seed=seed)
            row.update({"delta": float(np.mean(deltas)), "macro_delta": delta, "ci": [lo, hi],
                        "p": (one_sided_sign_flip_p([d + t.margin for d in deltas], seed=seed) if t.margin is not None
                              else sign_flip_p_value(deltas, seed=seed))})
        else:
            row.update({"delta": math.nan, "macro_delta": math.nan, "ci": [math.nan, math.nan], "p": math.nan})
        results.append(row)
    eligible = {r["id"]: r["p"] for r in results if r["k"] >= MIN_UNITS_FOR_VERDICT and r["margin"] is None}
    holm = holm_adjust(eligible)
    for r in results:
        if r["k"] < MIN_UNITS_FOR_VERDICT:
            r["holds"], r["verdict"] = None, f"none: K = {r['k']} < {MIN_UNITS_FOR_VERDICT}"
        elif r["margin"] is not None:
            r["holds"] = bool(r["p"] < NI_ALPHA)
            r["verdict"] = (f"non-inferior (one-sided p = {r['p']:.4f} < {NI_ALPHA}, margin {r['margin']})" if r["holds"]
                            else f"not shown non-inferior (one-sided p = {r['p']:.4f}, margin {r['margin']})")
        else:
            r["holm_p"] = holm[r["id"]]
            r["holds"] = bool(holm[r["id"]] < 0.05)
            r["verdict"] = ("significant" if r["holds"] else "not significant") + f" (Holm p = {holm[r['id']]:.4f})"
    return results


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x: float | None) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _oracle(us: Mapping[str, Mapping[str, Any]], key: str) -> float:
    """Macro IoU over units at the best single grid threshold (not deployable)."""
    best = math.nan
    n = max((len(u["models"][key]) for u in us.values() if key in u["models"]), default=0)
    for i in range(1, n):
        ious = [x for x in (_cm(u["models"][key][i]).iou for u in us.values() if key in u["models"]) if math.isfinite(x)]
        if ious:
            best = float(np.mean(ious)) if not math.isfinite(best) else max(best, float(np.mean(ious)))
    return best


def _test_lines(tests: Sequence[Mapping[str, Any]], names: Mapping[str, Any]) -> list[str]:
    lines = ["| Test | Question | B − A | Subset | K | Mean Δ IoU [95 % paired CI of the macro Δ] | B > A in | p | "
             "Verdict |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in tests:
        lines.append(f"| {r['id']} | {r['question']} | {names[r['b']]['name']} − {names[r['a']]['name']} | "
                     f"{r['subset']} | {r['k']} | {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
                     f"{r['wins']}/{r['k']} | {_fmt(r['p'])} | {r['verdict']} |")
    return lines


def _supported(by_unit: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    lines = ["| Test | Activations | Clusters | Supported (holds at both units) |", "| --- | --- | --- | --- |"]
    acts = {r["id"]: r for r in by_unit["activation"]}
    clus = {r["id"]: r for r in by_unit["cluster"]}
    for tid, a in acts.items():
        c = clus[tid]
        same_sign = (a["delta"] > 0) == (c["delta"] > 0)
        negative = a["margin"] is None and a["delta"] < 0 and c["delta"] < 0
        # A two-sided test significant at both units *against* B refutes its hypothesis (protocol_v1_1.md §4).
        verdict = ("no: refuted, A > B at both units" if a["holds"] and c["holds"] and negative else
                   "yes" if a["holds"] and c["holds"] and (same_sign or a["margin"] is not None) else
                   "no verdict" if a["holds"] is None or c["holds"] is None else "no")
        lines.append(f"| {tid} | {a['verdict']} | {c['verdict']} | {verdict} |")
    return lines


def report(scores: Mapping[str, Any], spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Write ``out`` (Markdown) and ``out.with_suffix('.json')`` from the scores alone."""
    from floodfusion.metrics import basin_bootstrap_ci, macro_iou
    from floodfusion.runs import write_json

    names = scores["models"]
    primary = spec["primary_stratum"]
    by_unit = {u: run_tests(units(scores, spec, stratum=primary, unit=u), spec) for u in ("activation", "cluster")}
    acts = units(scores, spec, stratum=primary)
    clusters = units(scores, spec, stratum=primary, unit="cluster")
    lines = [
        "# CEMS benchmark — independent flood maps, selected by a rule fixed in advance", "",
        "Zero-shot: no model saw any of these events; each uses the τ* and calibration of its own training data "
        "(for a model with an optical input, its SAR-only calibration wherever a window had no usable optical). "
        f"Primary stratum: products drawn on **{primary}** satellites; excluded: {', '.join(spec['exclude']) or 'none'}. "
        f"Units: {len(acts)} activations in {len(clusters)} clusters (within {CLUSTER_KM:.0f} km and "
        f"{CLUSTER_DAYS} days). Protocol: `reports/protocol_v1_1.md` (amendment 1). Sign-flip p-values are exact "
        f"for K ≤ 20 and Monte Carlo ({_MC_PATTERNS:,} patterns) above. Products: © European Union, Copernicus "
        "Emergency Management Service.", "",
        "## Pre-registered tests — does each hold at both unit definitions?", "", *_supported(by_unit), "",
        "### Per activation", "", *_test_lines(by_unit["activation"], names), "",
        "### Per cluster", "", *_test_lines(by_unit["cluster"], names), "",
    ]
    descriptive: dict[str, Any] = {}
    sensitivity = {
        "partial cloud (10-50 % clear), descriptive": dict(spec, tests=[replace(t, subset="partial", margin=None)
                                                                         for t in spec["tests"]]),
    }
    for title, sub_spec in sensitivity.items():
        res = run_tests(acts, sub_spec)
        descriptive[title] = res
        lines += [f"### {title.capitalize()}", "", *_test_lines(res, names), ""]
    for title, kwargs in ((f"without units holding < {MIN_FLOOD_KM2:g} km² of flood", {"min_flood_km2": MIN_FLOOD_KM2}),
                          ("without the pairs that overlap Kuro Siwo training areas",
                           {"exclude_pairs": spec["overlapping_pairs"]})):
        res = run_tests(units(scores, spec, stratum=primary, **kwargs), spec)
        descriptive[title] = res
        lines += [f"### Sensitivity: {title} (descriptive)", "", *_test_lines(res, names), ""]
    lines += ["## Every model, primary stratum, per activation (macro IoU at τ* [95 % CI]; oracle threshold)", "",
              "| Model | Role | IoU at τ* | IoU, oracle τ | K |", "| --- | --- | --- | --- | --- |"]
    summary: dict[str, Any] = {}
    for key, meta in names.items():
        per = {a: _cm(u["models"][key][0]) for a, u in acts.items() if key in u["models"]}
        per = {a: cm for a, cm in per.items() if math.isfinite(cm.iou)}
        if not per:
            continue
        ci = basin_bootstrap_ci(per, statistic="macro") if len(per) > 1 else (math.nan, math.nan, math.nan)
        summary[key] = {"iou": macro_iou(per), "ci": [ci[1], ci[2]], "oracle": _oracle(acts, key), "k": len(per)}
        lines.append(f"| {meta['name']} | {meta.get('role', 'test')} | {_fmt(summary[key]['iou'])} "
                     f"[{_fmt(summary[key]['ci'][0])}, {_fmt(summary[key]['ci'][1])}] | {_fmt(summary[key]['oracle'])} | "
                     f"{summary[key]['k']} |")
    everything = units(scores, {**spec, "exclude": ()}, stratum=None)
    cluster_of = activation_clusters(scores["activations"]) if scores.get("activations") else {}
    strata: dict[str, str] = {}
    for e in scores["pairs"].values():
        a, s = e["pair"]["activation"], e["pair"]["stratum"]
        strata[a] = s if strata.get(a, s) == s else "mixed (per pair)"
    lines += ["", "## Activations", "",
              "| Activation | Cluster | Stratum | Scored km² | Flood km² | Clear optical |", "| --- | --- | --- | --- | --- | --- |"]
    for a, u in sorted(everything.items()):
        lines.append(f"| {a} | {cluster_of.get(a, a)} | {strata[a]} | {u['scored_px'] * _PX_KM2:.0f} | "
                     f"{u['flood_px'] * _PX_KM2:.1f} | {u['clear_px'] / max(u['scored_px'], 1):.2f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"tests": by_unit, "descriptive": descriptive, "models": summary,
              "units": {"activation": sorted(acts), "cluster": sorted(clusters)}}
    write_json(out.with_suffix(".json"), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score models on the CEMS benchmark; report the pre-registered tests.")
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models.yaml"))
    sc.add_argument("--case-root", type=Path, default=Path("data/cems_benchmark"))
    sc.add_argument("--records", type=Path, default=Path("data/cems_records"))
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

        write_json(args.out, {**score(spec, args.case_root, args.device, records=args.records), "git": git_state()})
        return 0
    report(json.loads(args.scores.read_text(encoding="utf-8")), spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
