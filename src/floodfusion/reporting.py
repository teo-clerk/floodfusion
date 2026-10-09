r"""The ablation table of ``MILESTONES.md`` section 6, regenerated from run directories.

``python -m floodfusion.evaluate --ablation-table --runs-root runs --out reports/ablation_table.md``

Every number comes from the ``eval_test.json`` / ``run.json`` files the single evaluation
code path wrote; nothing is re-typed. The builder

1. discovers the runs and **refuses** to emit a table if any run was made from a dirty tree, or
   if the code that determines the numbers (``src/``, ``configs/``, ``pyproject.toml``,
   ``uv.lock``, ``Makefile``) differs between the run's SHA and the current tree — unless
   ``--allow-dirty`` (``P1-M6.3``). Documentation-only commits keep runs valid;
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
    holm_adjust,
    macro_iou,
    paired_bootstrap_delta,
    sign_flip_p_value,
)
from floodfusion.runs import RUN_RECORD, write_json

__all__ = ["GAPS", "MIN_BASINS_FOR_VERDICT", "ROW_LABELS", "build_ablation_table", "check_reproducibility",
           "cm_from_dict", "collect_rows", "fmt", "paired_gaps", "per_basin_table", "row_summary"]

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
#: Below this many test basins a percentile basin bootstrap undercovers badly (simulated
#: coverage ~0.80 at K = 4 for a nominal 0.95; false "resolvable" rate ~0.2 instead of 0.05),
#: so no verdict is issued: per-basin differences and their sign counts are reported instead.
MIN_BASINS_FOR_VERDICT = 8


def cm_from_dict(d: Mapping[str, Any]) -> ConfusionMatrix:
    return ConfusionMatrix(int(d["tp"]), int(d["fp"]), int(d["fn"]), int(d["tn"]))


def fmt(x: float | None, digits: int = 3) -> str:
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


def check_reproducibility(rows: Mapping[str, Sequence[Mapping[str, Any]]], allow_dirty: bool) -> dict[str, Any]:
    """Refuse a table that does not regenerate from the current code (``P1-M6.3``).

    Checked: every training run AND every evaluation was made from a clean tree whose code
    (``env.CODE_PATHS``) equals the current tree's — a later documentation or report commit does
    not invalidate a number, a code change does — plus the cross-run invariants a pooled table
    silently depends on: one manifest; one test set for every row except row 6 (cloud-free
    subset by definition); and, within a row, one configuration up to the seed.
    """
    from floodfusion.utils import env

    current = env.git_state()
    problems: list[str] = []
    if current.get("git_dirty"):
        problems.append("the working tree has uncommitted code/config changes")
    checked: dict[str | None, bool] = {}

    def code_ok(sha: str | None) -> bool:
        if sha not in checked:
            checked[sha] = sha == current.get("git_sha") or not env.code_changed_since(sha)
        return checked[sha]

    manifests, test_sets = set(), set()
    for key, entries in rows.items():
        configs = {e["run"].get("config_sha256_without_seed") for e in entries} - {None}
        if len(configs) > 1:
            problems.append(f"row {key}: seeds were trained with different configurations")
        for e in entries:
            run, ev = e["run"], e["eval"]
            if run.get("git_dirty") or not code_ok(run.get("git_sha")):
                problems.append(f"{e['dir']}: training at git {str(run.get('git_sha'))[:10]}"
                                f"{' (dirty)' if run.get('git_dirty') else ' (code differs)'}")
            eg = ev.get("evaluation_git") or {}
            if eg and (eg.get("git_dirty") or not code_ok(eg.get("git_sha"))):
                problems.append(f"{e['dir']}: evaluation at git {str(eg.get('git_sha'))[:10]} (dirty or code differs)")
            manifests.add(ev.get("manifest_sha256") or run.get("manifest_sha256"))
            if key != "6" and ev.get("test_chips_sha256"):
                test_sets.add(ev["test_chips_sha256"])
    if len(manifests - {None}) > 1:
        problems.append(f"runs from {len(manifests - {None})} different manifests are mixed")
    if len(test_sets) > 1:
        problems.append(f"rows were evaluated on {len(test_sets)} different test sets (row 6 excepted)")
    if problems and not allow_dirty:
        raise ValueError("refusing to emit a table that does not regenerate from one SHA (P1-M6.3):\n  "
                         + "\n  ".join(problems[:20]) + "\n(pass --allow-dirty to override)")
    return {**current, "reproducibility_problems": problems}


def _seed_of(entry: Mapping[str, Any]) -> int | None:
    seed = entry["run"].get("seed")
    return int(seed) if isinstance(seed, int) else None


def _pool_seeds(entries: Sequence[Mapping[str, Any]], seeds: set[int | None] | None = None) -> dict[str, ConfusionMatrix]:
    pooled: dict[str, ConfusionMatrix] = {}
    for e in entries:
        if seeds is not None and _seed_of(e) not in seeds:
            continue
        for basin, d in e["eval"]["per_basin"].items():
            cm = cm_from_dict(d)
            pooled[basin] = pooled[basin] + cm if basin in pooled else cm
    return pooled


def row_summary(key: str, entries: Sequence[Mapping[str, Any]], n_boot: int, alpha: float) -> dict[str, Any]:
    pooled = _pool_seeds(entries)
    _, lo, hi = basin_bootstrap_ci(pooled, n_boot=n_boot, alpha=alpha, seed=0, statistic="macro")
    f1s = [m.f1 for m in pooled.values() if not math.isnan(m.f1)]
    seeds = [e["eval"]["macro_iou"] for e in entries]
    trained = [e for e in entries if e["run"].get("trained", True) and e["run"].get("train_seconds")]
    latency = next((e["eval"]["latency"] for e in entries if e["eval"].get("latency")), None)
    vram_err = [e["run"]["vram_prediction"]["relative_error"] for e in trained
                if (e["run"].get("vram_prediction") or {}).get("relative_error") is not None]
    first = entries[0]["eval"]
    subsets = [e["eval"].get("cloud_free_subset") for e in entries if e["eval"].get("cloud_free_subset")]
    return {
        "key": key, "n_seeds": len(entries), "seeds": sorted(s for s in map(_seed_of, entries) if s is not None),
        "n_basins": len(pooled), "per_basin": pooled, "iou": macro_iou(pooled),
        "ci": (lo, hi) if len(pooled) >= 2 else (float("nan"), float("nan")),
        "cloud_free_iou": float(np.mean([c["macro_iou"] for c in subsets])) if subsets else None,
        "cloud_free_chips": subsets[0]["n_chips"] if subsets else None,
        "cloud_free_sha": subsets[0]["chips_sha256"] if subsets else None,
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
        mark = "†" if s["n_basins"] < MIN_BASINS_FOR_VERDICT else ""
        iou = f"{fmt(s['iou'])} [{fmt(s['ci'][0], 2)}, {fmt(s['ci'][1], 2)}]{mark}"
        vram = "—" if not s["trained"] else (f"{s['peak_vram_gib']:.2f} GiB" if s["peak_vram_gib"] else "—")
        train_h = "—" if not s["trained"] else fmt(s["train_h"], 3)
        if row == 11:
            vram, train_h = "(row 9)", "(row 9)"
        ms = fmt(s["latency_ms"], 1) + (" (CPU)" if s["latency_device"] == "cpu" and s["latency_ms"] else "")
        lines.append(f"| {row} | {config} | {purpose} | {iou} | {fmt(s['f1'], 3)} | {vram} | {train_h} | {ms} |")
    return lines


def _supplementary(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| Row | Seeds | IoU per seed (mean ± sd) | Pooled IoU | IoU raw σ(z)>τ* | IoU val-selected τ | "
             "IoU oracle τ | AP | ECE test raw → cal | Calibration | VRAM prediction error |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, s in sorted(summaries.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        sd = "" if not math.isfinite(s["seed_sd"]) else f" ± {s['seed_sd']:.3f}"
        err = "—" if s["vram_error"] is None else f"{100 * s['vram_error']:+.1f} %"
        ece = "—" if s["baseline"] else f"{fmt(s['ece_raw'], 4)} → {fmt(s['ece_cal'], 4)}"
        seeds = ", ".join(map(str, s["seeds"])) or "—"
        lines.append(f"| {key} | {seeds} | {fmt(s['seed_mean'])}{sd} | {fmt(s['pooled_iou'])} | "
                     f"{fmt(s['raw_tau'])} | {fmt(s['val_selected'])} | {fmt(s['oracle'])} | {fmt(s['ap'])} | "
                     f"{ece} | {', '.join(s['calibration'])} | {err} |")
    return lines


def paired_gaps(summaries: Mapping[str, Mapping[str, Any]], rows: Mapping[str, Sequence[Mapping[str, Any]]],
          n_boot: int, alpha: float,
          gaps: Sequence[tuple[Any, Any, str, str]] = GAPS) -> tuple[list[str], dict[str, Any]]:
    """Paired gaps on MATCHED seeds; a verdict only with enough basins to support one.

    The verdict is an exact sign-flip permutation test of the per-basin differences, Holm-
    adjusted across every gap of the table (family-wise error 0.05); the percentile basin
    bootstrap interval is reported as a description of the spread, not as the test.
    """
    out: dict[str, Any] = {}
    texts: dict[str, tuple[str, str, str, str]] = {}
    for a, b, question, expected in gaps:
        ra, rb = rows.get(str(a)), rows.get(str(b))
        if ra is None or rb is None or a in _NOT_COMPARABLE or b in _NOT_COMPARABLE:
            continue
        seeds_a, seeds_b = {_seed_of(e) for e in ra}, {_seed_of(e) for e in rb}
        # A seedless row (baseline) pairs with any seed set; otherwise only common seeds are compared,
        # so a 3-seed row is never set against a 1-seed row (the seed noise would differ).
        common = seeds_b if seeds_a == {None} else seeds_a if seeds_b == {None} else seeds_a & seeds_b
        if not common:
            continue
        pa = _pool_seeds(ra, None if seeds_a == {None} else common)
        pb = _pool_seeds(rb, None if seeds_b == {None} else common)
        key = f"{b}-{a}"
        if set(pa) != set(pb):
            texts[key] = (f"{b} − {a}", question, "", expected)
            out[key] = {"comparable": False}
            continue
        delta, lo, hi = paired_bootstrap_delta(pa, pb, n_boot=n_boot, alpha=alpha, seed=0, statistic="macro")
        per_basin = {k: pb[k].iou - pa[k].iou for k in pa if not (math.isnan(pa[k].iou) or math.isnan(pb[k].iou))}
        seeds_txt = ", ".join(str(x) for x in sorted(s for s in common if s is not None)) or "—"
        texts[key] = (f"{b} − {a}", question, seeds_txt, expected)
        out[key] = {"comparable": True, "delta": delta, "ci": [lo, hi], "basins_b_better": sum(d > 0 for d in per_basin.values()),
                    "n_basins": len(per_basin), "per_basin_delta": per_basin,
                    "p_value": sign_flip_p_value(list(per_basin.values())),
                    "seeds": sorted(s for s in common if s is not None)}
    holm = holm_adjust({k: g["p_value"] for k, g in out.items() if g.get("comparable") and g["n_basins"] >= MIN_BASINS_FOR_VERDICT})
    lines = ["| Gap (B − A) | Question | Seeds | Δ IoU [95 % paired CI] | B > A in | p (sign-flip) | Verdict | Expected sign |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, (label, question, seeds_txt, expected) in texts.items():
        g = out[key]
        if not g["comparable"]:
            lines.append(f"| {label} | {question} | | not comparable (different basins) | | | | {expected} |")
            continue
        if g["n_basins"] < MIN_BASINS_FOR_VERDICT:
            verdict, g["resolvable"], g["p_holm"] = f"none: K = {g['n_basins']} < {MIN_BASINS_FOR_VERDICT} basins", None, None
        else:
            g["p_holm"] = holm[key]
            g["resolvable"] = holm[key] < 0.05
            verdict = (f"significant (Holm p = {holm[key]:.3f})" if g["resolvable"]
                       else f"not significant (Holm p = {holm[key]:.3f})")
        lines.append(f"| {label} | {question} | {seeds_txt} | {g['delta']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] | "
                     f"{g['basins_b_better']}/{g['n_basins']} basins | {g['p_value']:.3f} | {verdict} | {expected} |")
    return lines, out


def per_basin_table(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    basins = sorted({b for s in summaries.values() for b in s["per_basin"]})
    lines = ["| Row | " + " | ".join(basins) + " |", "| --- |" + " --- |" * len(basins)]
    for key, s in sorted(summaries.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        cells = [fmt(s["per_basin"][b].iou) if b in s["per_basin"] else "—" for b in basins]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return lines


def _cloud_free(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every row on row 6's own test set (cloud-free chips), so the optical ceiling has comparators."""
    have = {k: s for k, s in summaries.items() if s.get("cloud_free_iou") is not None}
    if not have:
        return []
    reference = summaries.get("6", {}).get("cloud_free_sha")
    lines = ["| Row | Macro IoU at τ* on the cloud-free test chips | Chips | Same chips as row 6? |",
             "| --- | --- | --- | --- |"]
    for key, s in sorted(have.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        same = "—" if reference is None else ("yes" if s["cloud_free_sha"] == reference else "no")
        lines.append(f"| {key} | {fmt(s['cloud_free_iou'])} | {s['cloud_free_chips']} | {same} |")
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
        lines.append(f"| {name.replace('_', ' ')} | {fmt(v['macro_iou'])} | {v['macro_iou'] - ref:+.3f} | "
                     f"{fmt(v['macro_precision'])} | {fmt(v['macro_recall'])} |")
    return lines


def _statements(gaps: Mapping[str, Any]) -> list[str]:
    def say(key: str, text: str) -> str | None:
        g = gaps.get(key)
        if g is None or not g.get("comparable", True):
            return None
        if g["resolvable"] is None:
            verdict = (f"higher in {g['basins_b_better']} of {g['n_basins']} test basins; with this few basins no "
                       "verdict is possible")
        else:
            verdict = (f"significant (Holm-adjusted sign-flip p = {g['p_holm']:.3f})" if g["resolvable"]
                       else f"not significant at this number of basins (Holm p = {g['p_holm']:.3f})")
        return f"- {text}: **{g['delta']:+.3f}** [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] — {verdict}."

    candidates = [say("9-11", "Row 9 − row 11 (the honest operational expectation: the cost of losing optical)"),
                  say("13-9", "Row 13 − row 9 (the size of the spatial-leakage illusion)"),
                  say("9-2", "Row 9 − row 2 (what the machine learning was actually worth)")]
    return [c for c in candidates if c]


def kv_is_row(key: str) -> bool:
    return key.isdigit() and int(key) in ROW_LABELS


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
    git = check_reproducibility(rows, allow_dirty)
    summaries = {k: row_summary(k, v, n_boot, alpha) for k, v in rows.items()}
    gap_lines, gaps = paired_gaps(summaries, rows, n_boot, alpha)
    protocol_path = Path(runs_root) / "ablation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else None
    any_eval = next(iter(rows.values()))[0]
    synthetic = any(e["eval"].get("synthetic_data") or e["run"].get("synthetic_data") for v in rows.values() for e in v)
    test_basins = sorted({b for s in summaries.values() if not s["baseline"] for b in s["per_basin"]})
    seeds = ", ".join(f"row {k}: {len(s['seeds'])}" for k, s in sorted(summaries.items(), key=lambda kv: kv[0].zfill(3))
                      if s["trained"] and kv_is_row(k))
    protocol_text = (f"seed protocol **{protocol['protocol']}** (decided before the grid from a measured row-9 run: "
                     f"projected full grid {protocol['projected_full_grid_minutes']:.0f} min vs budget "
                     f"{protocol['budget_minutes']:.0f} min)" if protocol else "seed protocol not recorded")
    header = [
        "# Ablation table — MILESTONES.md section 6",
        "",
        *(["> **SYNTHETIC BENCHMARK.** Every number below is measured on the physics-based synthetic "
           "benchmark (`floodfusion.data.synthetic`), not on real Sentinel-1/2 data. It validates the "
           "pipeline and the experimental design; it is **not** a real-world accuracy claim.", ""] if synthetic else []),
        f"**Protocol** (docs/09 §2.2): one fixed basin-blocked train/val/test split; "
        f"test basins: {', '.join(test_basins)} ({len(test_basins)}); {protocol_text}; seeds per learned row — "
        f"{seeds or '—'}; gaps compare **matched seeds only**. **IoU** = macro over test basins at the cost-optimal τ* = 1/(1+r) = "
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
        "of split** (docs/09 §2.2). † Fewer than "
        f"{MIN_BASINS_FOR_VERDICT} test basins: a percentile bootstrap over so few basins undercovers (simulated "
        "coverage ≈ 0.80 at K = 4 for a nominal 0.95), so the intervals are **indicative only** and the gaps below "
        "carry no verdict — the per-basin direction is reported instead. With 8 or more basins the verdict is an exact "
        "sign-flip permutation test of the per-basin differences, Holm-adjusted across the gaps.", "",
        "## Beneath the table: the gaps that matter (paired basin bootstrap, docs/09 §3.3 and §5.1)", "",
        *gap_lines, "", *_statements(gaps), "",
        "Row 6 is scored on cloud-free chips only — a different test set — and is excluded from every paired "
        "comparison; its comparators on the same chips are below.", "",
        *(["## Row 6's test set: every row on the cloud-free test chips", "", *_cloud_free(summaries), ""]
          if _cloud_free(summaries) else []),
        "## Supplementary", "", *_supplementary(summaries), "",
        "## Per-basin IoU at τ* (seeds pooled)", "", *per_basin_table(summaries), "",
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
            "protocol": protocol,
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
