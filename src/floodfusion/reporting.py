r"""The ablation table of ``MILESTONES.md`` section 6, regenerated from run directories.

``python -m floodfusion.evaluate --ablation-table --runs-root runs --out reports/ablation_table.md``

Every number comes from the ``eval_test.json`` / ``run.json`` files the single evaluation
code path wrote; nothing is re-typed. The builder

1. discovers the runs and **refuses** to emit a table if any run's git SHA differs from the
   current tree, or the tree is dirty — unless ``--allow-dirty`` (``P1-M6.3``);
2. pools each learned row's seeds **per basin** (confusion counts summed over seeds), reports
   macro-over-basin IoU at :math:`\tau^\star` with a 95 % percentile bootstrap over test
   basins, and the seed spread separately (basin-to-basin and seed-to-seed variance are
   different quantities, ``docs/09`` section 3.4);
3. emits the main table with exactly the columns of ``MILESTONES.md`` section 6, then a
   supplementary table, the **paired** basin-bootstrap gaps of ``docs/09`` section 5.1 (one
   shared resample for both rows, ``docs/09`` section 3.3), per-basin IoU, the baseline mask
   attribution, and the factual statements beneath the table;
4. draws the per-basin dot plot (``docs/09`` section 8.2, figure 4).

Row 6 is scored on cloud-free chips only and is excluded from every paired comparison.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import (
    ConfusionMatrix,
    basin_bootstrap_ci,
    macro_iou,
    paired_bootstrap_delta,
)
from floodfusion.runs import RUN_RECORD, write_json

__all__ = ["GAPS", "ROW_LABELS", "build_ablation_table", "collect_rows"]

LOGGER = logging.getLogger(__name__)

#: MILESTONES.md section 6, verbatim: (configuration, purpose).
ROW_LABELS: dict[int, tuple[str, str]] = {
    1: ("Global Otsu on VV, no masks", "The floor. Is anything learned at all?"),
    2: ("Baseline: adaptive threshold + change + masks", "The operational reference you must beat"),
    3: ("Baseline + MNDWI where cloud-free", "What optical adds to a non-learned method"),
    4: ("U-Net, SAR only (VV, VH)", "Does learning beat thresholding on the same inputs?"),
    5: ("U-Net, SAR + DEM derivatives", "What terrain context is worth"),
    6: ("U-Net, optical only, cloud-free chips only", "The optical ceiling — an upper bound you cannot deploy"),
    7: ("Fusion, early (channel concatenation)", "Does naive fusion help?"),
    8: ("Fusion, late (decision-level)", "Where in the network fusion should happen"),
    9: ("**Fusion, cross-attention (reference)**", "The proposed model"),
    10: ("Row 9 − modality dropout", "**Is modality dropout load-bearing?**"),
    11: ("Row 9, evaluated SAR-only at inference", "**Graceful degradation — the operational case**"),
    12: ("Row 9 − compound loss (plain BCE)", "Does the imbalance objective matter?"),
    13: ("Row 9, random split instead of basin-blocked", "**The leakage row. Expect an inflated score.**"),
}

#: docs/09 section 5.1: (row A, row B, question, expected sign of B - A).
GAPS: tuple[tuple[int, int, str, str], ...] = (
    (1, 2, "What careful engineering adds to naive Otsu", "large positive"),
    (2, 4, "What learning adds over a good physics baseline", "positive, possibly modest"),
    (4, 5, "What terrain context is worth", "positive"),
    (7, 9, "Whether *where* you fuse matters (vs early)", "small"),
    (8, 9, "Whether *where* you fuse matters (vs late)", "small"),
    (11, 9, "The cost of losing optical (9 − 11)", "positive; small is good"),
    (10, 9, "Whether modality dropout is load-bearing (9 − 10)", "positive"),
    (12, 9, "What the compound loss was worth (9 − 12)", "positive"),
    (9, 13, "The size of the spatial-leakage illusion (13 − 9)", "positive — row 13 must be higher"),
    (4, 11, "Row 11 vs SAR-only from scratch (docs/09 5.2)", "≈ 0: optical never mattered; > 0: it helped training"),
    (2, 9, "What the machine learning was actually worth (9 − 2)", "positive"),
)
_NOT_COMPARABLE = {6}


def _cm(d: Mapping[str, Any]) -> ConfusionMatrix:
    return ConfusionMatrix(int(d["tp"]), int(d["fp"]), int(d["fn"]), int(d["tn"]))


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def collect_rows(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    """``row key -> [{"run": run.json, "eval": eval_test.json, "dir": path}, ...]`` (seeds)."""
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_path in sorted(Path(runs_root).glob(f"*/*/{RUN_RECORD}")):
        run_dir = record_path.parent
        eval_path = run_dir / "eval_test.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not eval_path.is_file():
            LOGGER.warning("%s has no eval_test.json — skipped", run_dir)
            continue
        result = json.loads(eval_path.read_text(encoding="utf-8"))
        row = result.get("ablation_row")
        key = str(int(row)) if isinstance(row, int) and row in ROW_LABELS else str(result.get("name", record["name"]))
        rows[key].append({"run": record, "eval": result, "dir": run_dir})
    return dict(rows)


def _check_reproducibility(rows: Mapping[str, Sequence[Mapping[str, Any]]], allow_dirty: bool) -> dict[str, Any]:
    from floodfusion.utils.env import git_state

    current = git_state()
    problems = []
    if current.get("git_dirty"):
        problems.append("the working tree is dirty")
    for entries in rows.values():
        for e in entries:
            sha, dirty = e["run"].get("git_sha"), e["run"].get("git_dirty")
            if sha != current.get("git_sha") or dirty:
                problems.append(f"{e['dir']}: git {str(sha)[:10]}{' (dirty)' if dirty else ''}")
    if problems and not allow_dirty:
        raise ValueError("refusing to emit a table that does not regenerate from one SHA (P1-M6.3):\n  "
                         + "\n  ".join(problems[:20]) + "\n(pass --allow-dirty to override)")
    return {**current, "reproducibility_problems": problems}


def _pool_seeds(entries: Sequence[Mapping[str, Any]]) -> dict[str, ConfusionMatrix]:
    pooled: dict[str, ConfusionMatrix] = {}
    for e in entries:
        for basin, d in e["eval"]["per_basin"].items():
            cm = _cm(d)
            pooled[basin] = pooled[basin] + cm if basin in pooled else cm
    return pooled


def _row_summary(key: str, entries: Sequence[Mapping[str, Any]], n_boot: int, alpha: float) -> dict[str, Any]:
    pooled = _pool_seeds(entries)
    _, lo, hi = basin_bootstrap_ci(pooled, n_boot=n_boot, alpha=alpha, seed=0, statistic="macro")
    f1s = [m.f1 for m in pooled.values() if not math.isnan(m.f1)]
    seeds = [e["eval"]["macro_iou"] for e in entries]
    trained = [e for e in entries if e["run"].get("trained", True) and e["run"].get("train_seconds")]
    latency = next((e["eval"]["latency"] for e in entries if e["eval"].get("latency")), None)
    vram_err = [e["run"]["vram_prediction"]["relative_error"] for e in trained
                if (e["run"].get("vram_prediction") or {}).get("relative_error") is not None]
    first = entries[0]["eval"]
    return {
        "key": key, "n_seeds": len(entries), "per_basin": pooled, "iou": macro_iou(pooled), "ci": (lo, hi),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "seed_mean": float(np.mean(seeds)), "seed_sd": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan"),
        "peak_vram_gib": max((e["run"].get("peak_vram_bytes", 0) for e in trained), default=0) / 1024**3 or None,
        "train_h": float(np.mean([e["run"]["train_seconds"] for e in trained])) / 3600 if trained else None,
        "latency_ms": latency.get("latency_ms_median") if latency else None,
        "latency_device": (latency or {}).get("device", "cuda"),
        "pooled_iou": first.get("pooled_iou"), "raw_tau": _mean(entries, "macro_iou_raw_tau_star"),
        "val_selected": _mean(entries, "macro_iou_val_selected"), "oracle": _mean(entries, "macro_iou_oracle"),
        "ap": _mean(entries, "average_precision"), "ece_raw": _mean(entries, "ece_test_raw"),
        "ece_cal": _mean(entries, "ece_test_calibrated"),
        "calibration": sorted({e["eval"].get("calibration", {}).get("method", "—") for e in entries}),
        "vram_error": float(np.mean(vram_err)) if vram_err else None,
        "trained": bool(trained), "baseline": bool(first.get("baseline")),
    }


def _mean(entries: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [e["eval"].get(key) for e in entries]
    finite = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _main_table(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| #   | Configuration | Purpose — the question the row answers | IoU | F1  | Peak VRAM | Train h | ms/chip |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row, (config, purpose) in ROW_LABELS.items():
        s = summaries.get(str(row))
        if s is None:
            lines.append(f"| {row} | {config} | {purpose} | not run | | | | |")
            continue
        iou = f"{_fmt(s['iou'])} [{_fmt(s['ci'][0], 2)}, {_fmt(s['ci'][1], 2)}]"
        vram = "—" if not s["trained"] else (f"{s['peak_vram_gib']:.2f} GiB" if s["peak_vram_gib"] else "—")
        train_h = "—" if not s["trained"] else _fmt(s["train_h"], 3)
        if row == 11:
            vram, train_h = "(row 9)", "(row 9)"
        ms = _fmt(s["latency_ms"], 1) + (" (CPU)" if s["latency_device"] == "cpu" and s["latency_ms"] else "")
        lines.append(f"| {row} | {config} | {purpose} | {iou} | {_fmt(s['f1'], 3)} | {vram} | {train_h} | {ms} |")
    return lines


def _supplementary(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| Row | Seeds | IoU per seed (mean ± sd) | Pooled IoU | IoU raw σ(z)>τ* | IoU val-selected τ | "
             "IoU oracle τ | AP | ECE test raw → cal | Calibration | VRAM prediction error |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, s in sorted(summaries.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        sd = "" if not math.isfinite(s["seed_sd"]) else f" ± {s['seed_sd']:.3f}"
        err = "—" if s["vram_error"] is None else f"{100 * s['vram_error']:+.1f} %"
        ece = "—" if s["baseline"] else f"{_fmt(s['ece_raw'], 4)} → {_fmt(s['ece_cal'], 4)}"
        lines.append(f"| {key} | {s['n_seeds']} | {_fmt(s['seed_mean'])}{sd} | {_fmt(s['pooled_iou'])} | "
                     f"{_fmt(s['raw_tau'])} | {_fmt(s['val_selected'])} | {_fmt(s['oracle'])} | {_fmt(s['ap'])} | "
                     f"{ece} | {', '.join(s['calibration'])} | {err} |")
    return lines


def _gaps(summaries: Mapping[str, Mapping[str, Any]], n_boot: int, alpha: float) -> tuple[list[str], dict[str, Any]]:
    lines = ["| Gap (B − A) | Question | Δ IoU [95 % paired CI] | Resolvable? | Expected sign |",
             "| --- | --- | --- | --- | --- |"]
    out: dict[str, Any] = {}
    for a, b, question, expected in GAPS:
        sa, sb = summaries.get(str(a)), summaries.get(str(b))
        if sa is None or sb is None or a in _NOT_COMPARABLE or b in _NOT_COMPARABLE:
            continue
        if set(sa["per_basin"]) != set(sb["per_basin"]):
            lines.append(f"| {b} − {a} | {question} | not comparable (different basins) | | {expected} |")
            continue
        delta, lo, hi = paired_bootstrap_delta(sa["per_basin"], sb["per_basin"], n_boot=n_boot, alpha=alpha,
                                               seed=0, statistic="macro")
        resolvable = "yes" if lo > 0 or hi < 0 else "no (CI contains 0)"
        lines.append(f"| {b} − {a} | {question} | {delta:+.3f} [{lo:+.3f}, {hi:+.3f}] | {resolvable} | {expected} |")
        out[f"{b}-{a}"] = {"delta": delta, "ci": [lo, hi], "resolvable": lo > 0 or hi < 0}
    return lines, out


def _per_basin(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    basins = sorted({b for s in summaries.values() for b in s["per_basin"]})
    lines = ["| Row | " + " | ".join(basins) + " |", "| --- |" + " --- |" * len(basins)]
    for key, s in sorted(summaries.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        cells = [_fmt(s["per_basin"][b].iou) if b in s["per_basin"] else "—" for b in basins]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return lines


def _attribution(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    entries = rows.get("2")
    attribution = entries[0]["eval"].get("mask_attribution") if entries else None
    if not attribution:
        return []
    ref = attribution["all_masks"]["macro_iou"]
    lines = ["| Variant of row 2 | Test macro IoU | Δ vs all masks | Macro precision | Macro recall |",
             "| --- | --- | --- | --- | --- |"]
    for name, v in attribution.items():
        lines.append(f"| {name.replace('_', ' ')} | {_fmt(v['macro_iou'])} | {v['macro_iou'] - ref:+.3f} | "
                     f"{_fmt(v['macro_precision'])} | {_fmt(v['macro_recall'])} |")
    return lines


def _statements(gaps: Mapping[str, Any]) -> list[str]:
    def say(key: str, text: str) -> str | None:
        g = gaps.get(key)
        if g is None:
            return None
        verdict = "resolvable at this number of basins" if g["resolvable"] else "NOT resolvable at this number of basins"
        return f"- {text}: **{g['delta']:+.3f}** [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] — {verdict}."

    candidates = [say("9-11", "Row 9 − row 11 (the honest operational expectation: the cost of losing optical)"),
                  say("13-9", "Row 13 − row 9 (the size of the spatial-leakage illusion)"),
                  say("9-2", "Row 9 − row 2 (what the machine learning was actually worth)")]
    return [c for c in candidates if c]


def build_ablation_table(
    runs_root: Path | str,
    *,
    out: Path | str | None = None,
    allow_dirty: bool = False,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> str:
    """Build (and optionally write) the Markdown table plus its JSON sidecar and figure."""
    from floodfusion.figures import plot_basin_dots

    rows = collect_rows(Path(runs_root))
    if not rows:
        raise FileNotFoundError(f"no evaluated runs under {runs_root}")
    git = _check_reproducibility(rows, allow_dirty)
    summaries = {k: _row_summary(k, v, n_boot, alpha) for k, v in rows.items()}
    gap_lines, gaps = _gaps(summaries, n_boot, alpha)
    any_eval = next(iter(rows.values()))[0]
    synthetic = any(e["eval"].get("synthetic_data") or e["run"].get("synthetic_data") for v in rows.values() for e in v)
    test_basins = sorted({b for s in summaries.values() if not s["baseline"] for b in s["per_basin"]})
    seeds = sorted({s["n_seeds"] for k, s in summaries.items() if s["trained"]})
    header = [
        "# Ablation table — MILESTONES.md section 6",
        "",
        *(["> **SYNTHETIC BENCHMARK.** Every number below is measured on the physics-based synthetic "
           "benchmark (`floodfusion.data.synthetic`), not on real Sentinel-1/2 data. It validates the "
           "pipeline and the experimental design; it is **not** a real-world accuracy claim.", ""] if synthetic else []),
        f"**Protocol** (docs/09 §2.2, the affordable protocol): one fixed basin-blocked train/val/test split; "
        f"test basins: {', '.join(test_basins)} ({len(test_basins)}); seeds per learned row: "
        f"{', '.join(map(str, seeds)) or '—'}. **IoU** = macro over test basins at the cost-optimal τ* = 1/(1+r) = "
        "0.25 (r = 3) after calibration on validation (prior correction + temperature, Platt if clearly better); "
        "seeds pooled per basin; **[CI]** = 95 % percentile bootstrap over test basins (10 000 resamples). "
        "Baselines (rows 1–3): threshold offset chosen on validation by the same cost rule. **Peak VRAM** = "
        "`torch.cuda.max_memory_allocated()` during training; **ms/chip** = 256² chip, batch 1, ≥ 20 warm-up "
        "iterations after a sustained warm-up, `synchronize()` around the timer (docs/09 §7).",
        "",
        f"Manifest `{any_eval['eval'].get('manifest_sha256', '')[:16]}…`, git `{str(git.get('git_sha'))[:12]}`"
        f"{' (DIRTY — not reproducible)' if git['reproducibility_problems'] else ''}. Regenerate with "
        f"`python -m floodfusion.evaluate --ablation-table --runs-root {runs_root}`.",
        "",
    ]
    body = [
        *header, *_main_table(summaries), "",
        "**Uncertainty from a single fixed split reflects basin-to-basin variation, not variation in the choice "
        "of split** (docs/09 §2.2). With this few test basins the intervals are wide; that is the honest "
        "statement of what the data can resolve.", "",
        "## Beneath the table: the gaps that matter (paired basin bootstrap, docs/09 §3.3 and §5.1)", "",
        *gap_lines, "", *_statements(gaps), "",
        "Row 6 is scored on cloud-free chips only — a different test set — and is excluded from every paired "
        "comparison.", "",
        "## Supplementary", "", *_supplementary(summaries), "",
        "## Per-basin IoU at τ* (seeds pooled)", "", *_per_basin(summaries), "",
    ]
    attribution = _attribution(rows)
    if attribution:
        body += ["## Baseline mask attribution (P1-M2.4, row 2 at its operating point, test)", "", *attribution, ""]
    text = "\n".join(body)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        write_json(out.with_suffix(".json"), {
            "rows": {k: {**{kk: vv for kk, vv in s.items() if kk != "per_basin"},
                         "per_basin": {b: {"tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn, "iou": m.iou}
                                       for b, m in s["per_basin"].items()}} for k, s in summaries.items()},
            "gaps": gaps, "git": git, "synthetic": synthetic,
        })
        plot_basin_dots([{"label": f"{k} {ROW_LABELS[int(k)][0].strip('*')[:38]}" if k.isdigit() else k[:46],
                          "per_basin": {b: m.iou for b, m in s["per_basin"].items()}, "iou": s["iou"], "ci": s["ci"]}
                         for k, s in summaries.items()],
                        out.parent / "figures" / "ablation_basin_dots.png",
                        title="Per-basin IoU at τ* by ablation row" + (" (SYNTHETIC)" if synthetic else ""))
        LOGGER.info("wrote %s", out)
    return text
