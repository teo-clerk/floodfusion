r"""v1.2's confirmatory analysis (``reports/protocol_v1_2.md`` §4–5), fixed in code before the freeze.

Run once, at the stopping rule (:func:`floodfusion.prospective.stopping_rule_met`):

    uv run python -m floodfusion.blind_analysis reveal      # verifies every sealed file, then writes the verdicts

**The data.** The sealed per-pair scores (``data/prospective/sealed``). Each one's sha256 must match the committed ledger
before anything is read. Merged, they are scored like the CEMS benchmark: :func:`floodfusion.cems_scores_v12.summary`
over the primary stratum, per activation and per cluster.

**The tests.** They are listed under ``blind_tests`` in the frozen spec (``configs/datasets/prospective_v12.yaml``).
Each one compares model B (v1.2's) with model A:
- **The metric:** the per-unit gap B − A in macro IoU at each model's own τ* (seeds pooled), as in v1.1. A spec field
  ``metric_b`` names B's operator if an inference add-on was adopted.
- **The test:** a two-sided exact sign-flip test.
- **P5** (``also_ap``) must also win threshold-free. Its two conditions form one intersection-union test,
  p = max(p_IoU, p_AP).
- **P6** (``gfm_covered``) uses only the units GFM covers on at least 95 % of their scored pixels.
- **Multiplicity:** Holm across the tests within each unit definition. A test with fewer than
  :data:`~floodfusion.cems_scores.MIN_UNITS_FOR_VERDICT` units gives no verdict.

**The verdicts.**
- *supported*: significant, with every gap positive, at both unit definitions;
- *refuted*: significant, with every gap negative, at both;
- *no verdict*: too few units at either;
- *not supported*: anything else.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["UNITS", "load_blind_tests", "main", "merge_sealed", "unit_tests", "verdicts"]

LOGGER = logging.getLogger(__name__)

UNITS = ("activation", "cluster")


def load_blind_tests(spec_path: Path) -> list[dict[str, Any]]:
    """The ``blind_tests`` of a frozen spec, validated against its models."""
    import yaml

    raw = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    tests = list(raw.get("blind_tests", []))
    keys = {m["key"] for m in raw["models"]}
    for t in tests:
        unknown = {t["a"], t["b"]} - keys
        if unknown:
            raise ValueError(f"blind test {t['id']} names unknown models {sorted(unknown)}")
    if not tests:
        raise ValueError(f"{spec_path} lists no blind_tests")
    return tests


def merge_sealed(ledger: Any, sealed_dir: Path) -> dict[str, Any]:
    """Every scored pair's sealed scores, verified against the ledger, merged into one scores document."""
    import hashlib

    problems = ledger.problems()
    merged: dict[str, Any] = {"pairs": {}, "activations": {}, "priors": {}, "models": {}}
    for key, row in sorted(ledger.state().items()):
        if row.get("status") != "scored":
            continue
        path = sealed_dir / f"{key}.json"
        blob = path.read_bytes() if path.is_file() else b""          # read once: what is hashed is what is parsed
        if hashlib.sha256(blob).hexdigest() != row["sealed_sha256"]:
            problems.append(f"{key}: sealed file missing or changed")
            continue
        part = json.loads(blob)
        if set(part["pairs"]) != {key} or set(part["activations"]) - {row["activation"]}:
            problems.append(f"{key}: the sealed file holds other pairs or activations")
            continue
        merged["pairs"].update(part["pairs"])
        merged["activations"].update(part["activations"])
        merged["priors"] = merged["priors"] or part["priors"]
        merged["models"] = merged["models"] or part["models"]
    if problems:
        raise ValueError("sealed scores do not match the ledger:\n  " + "\n  ".join(problems))
    return merged


def _gaps(summ: Mapping[str, Any], a: str, b: str, metric_a: str, metric_b: str) -> tuple[list[str], list[float]]:
    pa, pb = summ["models"][a]["per_unit"], summ["models"][b]["per_unit"]
    keys = sorted(k for k in set(pa) & set(pb) if math.isfinite(pa[k][metric_a]) and math.isfinite(pb[k][metric_b]))
    return keys, [pb[k][metric_b] - pa[k][metric_a] for k in keys]


def unit_tests(scores: Mapping[str, Any], spec: Mapping[str, Any], tests: Sequence[Mapping[str, Any]],
               unit: str, *, seed: int = 0) -> list[dict[str, Any]]:
    """Every blind test at one unit definition: gaps, p-values, Holm, and whether it is significant."""
    from floodfusion.cems_scores import MIN_UNITS_FOR_VERDICT
    from floodfusion.cems_scores_v12 import summary
    from floodfusion.metrics import holm_adjust, sign_flip_p_value

    tables = {False: summary(scores, spec, unit=unit), True: summary(scores, spec, unit=unit, gfm_common=True)}
    rows = []
    for t in tests:
        summ = tables[bool(t.get("gfm_covered"))]
        keys, gaps = _gaps(summ, t["a"], t["b"], "iou_tau", t.get("metric_b", "iou_tau"))
        row: dict[str, Any] = {"id": t["id"], "a": t["a"], "b": t["b"], "units": keys, "k": len(keys),
                               "delta": float(np.mean(gaps)) if gaps else math.nan,
                               "wins": int(sum(g > 0 for g in gaps)), "p_iou": sign_flip_p_value(gaps, seed=seed)}
        deltas, p = [row["delta"]], row["p_iou"]
        if t.get("also_ap"):
            keys_ap, gaps_ap = _gaps(summ, t["a"], t["b"], "ap", "ap")
            row.update(k_ap=len(keys_ap), delta_ap=float(np.mean(gaps_ap)) if gaps_ap else math.nan,
                       wins_ap=int(sum(g > 0 for g in gaps_ap)), p_ap=sign_flip_p_value(gaps_ap, seed=seed))
            row["k"] = min(row["k"], row["k_ap"])
            deltas.append(row["delta_ap"])
            p = max(p, row["p_ap"])                                  # intersection-union: both must hold
        row["p"] = p
        row["sign"] = 1 if all(d > 0 for d in deltas) else -1 if all(d < 0 for d in deltas) else 0
        rows.append(row)
    holm = holm_adjust({r["id"]: r["p"] for r in rows if r["k"] >= MIN_UNITS_FOR_VERDICT})
    for r in rows:
        r["holm_p"] = holm.get(r["id"], math.nan)
        r["significant"] = None if r["k"] < MIN_UNITS_FOR_VERDICT else bool(r["holm_p"] < 0.05)
    return rows


def verdicts(scores: Mapping[str, Any], spec: Mapping[str, Any], tests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per test: both unit definitions' rows and the verdict (module docstring)."""
    by_unit = {u: {r["id"]: r for r in unit_tests(scores, spec, tests, u)} for u in UNITS}
    out = {}
    for t in tests:
        rows = [by_unit[u][t["id"]] for u in UNITS]
        if any(r["significant"] is None for r in rows):
            verdict = "no verdict"
        elif all(r["significant"] and r["sign"] == 1 for r in rows):
            verdict = "supported"
        elif all(r["significant"] and r["sign"] == -1 for r in rows):
            verdict = "refuted"
        else:
            verdict = "not supported"
        out[t["id"]] = {"verdict": verdict, **dict(zip(UNITS, rows))}
    return out


def _markdown(result: Mapping[str, Any], tests: Sequence[Mapping[str, Any]], names: Mapping[str, Any]) -> str:
    def f(x: float, sign: bool = False) -> str:
        return "—" if not math.isfinite(x) else (f"{x:+.3f}" if sign else f"{x:.4f}")

    lines = ["# v1.2 blind test — the pre-registered verdicts", "",
             "Every eligible Copernicus EMS flood activation from EMSR899 on, scored by models frozen before it "
             "(`reports/protocol_v1_2.md`). Sealed scores verified against `reports/prospective/ledger.jsonl`.", "",
             "| Test | B − A | Unit | K | Mean gap (IoU at τ*) | B > A in | AP gap | Holm p | Verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in tests:
        res = result[t["id"]]
        for u in UNITS:
            r = res[u]
            lines.append(f"| {t['id']} | {names[t['b']]['name']} − {names[t['a']]['name']} | {u} | {r['k']} | "
                         f"{f(r['delta'], True)} | {r['wins']}/{len(r['units'])} | {f(r.get('delta_ap', math.nan), True)}"
                         f" | {f(r['holm_p'])} | {res['verdict'] if u == 'activation' else ''} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from floodfusion.prospective import Ledger, Paths

    parser = argparse.ArgumentParser(description="v1.2's confirmatory analysis, once, at the stopping rule.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("reveal", help="verify the sealed scores against the ledger, then write the verdicts")
    rv.add_argument("--spec", type=Path, default=Path("configs/datasets/prospective_v12.yaml"))
    rv.add_argument("--ledger", type=Path, default=Paths().ledger)
    rv.add_argument("--sealed", type=Path, default=Paths().sealed)
    rv.add_argument("--out", type=Path, default=Path("reports/v12_blind_verdicts.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from floodfusion.cems_scores import load_spec

    tests = load_blind_tests(args.spec)
    scores = merge_sealed(Ledger(args.ledger), args.sealed)
    result = verdicts(scores, load_spec(args.spec), tests)
    args.out.write_text(_markdown(result, tests, scores["models"]), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    LOGGER.info("verdicts: %s", {k: v["verdict"] for k, v in result.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
