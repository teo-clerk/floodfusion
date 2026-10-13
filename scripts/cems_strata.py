"""Every CEMS benchmark model on every stratum and optical subset — the descriptive table of protocol §5.

``reports/cems_benchmark.md`` lists every model on the primary stratum (products drawn on radar satellites) only.
This re-reads the stored confusion counts (``reports/cems_benchmark_scores.json``; nothing is re-scored) and adds the
products drawn on optical VHR satellites, EMSR664 (Forlì, seen in v1.0, never in a test) and, per stratum, the optical,
partial-cloud and SAR-only subsets. Descriptive: no verdict. Writes ``reports/cems_benchmark_strata.{md,json}``.

Run: ``uv run --frozen python scripts/cems_strata.py``
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from floodfusion.cems_scores import _PX_KM2, _cm, _fmt, _oracle, _subset, load_spec, units
from floodfusion.metrics import basin_bootstrap_ci, macro_iou

SUBSETS = (("all", "every unit"), ("optical", "≥ 50 % clear"), ("partial", "10–50 % clear"),
           ("sar_only", "< 10 % clear"))


def strata(scores: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{title: units}``: the radar-drawn and optical-drawn strata (EMSR664 excluded) and EMSR664 on its own."""
    excluded = tuple(spec["exclude"])
    everything = units(scores, {**spec, "exclude": ()}, stratum=None)
    return {
        "Drawn on radar satellites (the primary stratum)": units(scores, spec, stratum="sar"),
        "Drawn on optical VHR satellites": units(scores, spec, stratum="optical"),
        f"{', '.join(excluded)} (seen in v1.0; every stratum)": {a: u for a, u in everything.items() if a in excluded},
    }


def model_rows(us: Mapping[str, Mapping[str, Any]], names: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per model: macro IoU at τ* over the units, its 95 % bootstrap interval, the oracle-threshold IoU and K."""
    rows = []
    for key, meta in names.items():
        per = {a: _cm(u["models"][key][0]) for a, u in us.items() if key in u["models"]}
        per = {a: cm for a, cm in per.items() if math.isfinite(cm.iou)}
        if not per:
            continue
        ci = basin_bootstrap_ci(per, statistic="macro") if len(per) > 1 else (math.nan, math.nan, math.nan)
        rows.append({"key": key, "name": meta["name"], "role": meta.get("role", "test"), "iou": macro_iou(per),
                     "ci": [ci[1], ci[2]], "oracle": _oracle({a: us[a] for a in per}, key), "k": len(per)})
    return rows


def build(scores: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    names = {m.key: {"name": m.name, "role": m.role} for m in spec["models"]}
    lines = ["# CEMS benchmark — every model on every stratum and optical subset (descriptive)", "",
             "Protocol §5 (`reports/protocol_v1_1.md`): every model on every stratum, the optical and SAR-only subsets, "
             "the oracle-threshold IoU and K per subset. Re-read from the stored confusion counts of "
             "`reports/cems_benchmark_scores.json` by `scripts/cems_strata.py`; nothing is re-scored, and nothing here "
             "carries a verdict. Units are activations, AOIs and seeds pooled. A subset is chosen by the unit's share "
             "of scored pixels with clear optical. Products: © European Union, Copernicus Emergency Management "
             "Service.", ""]
    result: dict[str, Any] = {}
    for title, us in strata(scores, spec).items():
        flood = sum(u["flood_px"] for u in us.values()) * _PX_KM2
        lines += [f"## {title}", "", f"{len(us)} activations, {flood:.0f} km² of mapped flood.", ""]
        result[title] = {}
        for subset, label in SUBSETS:
            keys = _subset(us, subset, spec)
            if not keys:
                lines += [f"**{subset}** ({label}): no unit.", ""]
                continue
            rows = model_rows({k: us[k] for k in keys}, names)
            result[title][subset] = {"units": keys, "models": rows}
            lines += [f"**{subset}** ({label}): K = {len(keys)}." + (" Fewer than 8 units: indicative only."
                                                                       if len(keys) < 8 else ""), "",
                      "| Model | Role | IoU at τ* [95 % CI] | IoU, oracle τ | K |", "| --- | --- | --- | --- | --- |"]
            lines += [f"| {r['name']} | {r['role']} | {_fmt(r['iou'])} [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] | "
                      f"{_fmt(r['oracle'])} | {r['k']} |" for r in rows]
            lines.append("")
    return lines, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models.yaml"))
    parser.add_argument("--scores", type=Path, default=Path("reports/cems_benchmark_scores.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/cems_benchmark_strata.md"))
    args = parser.parse_args(argv)
    lines, result = build(json.loads(args.scores.read_text(encoding="utf-8")), load_spec(args.spec))
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
