r"""The real-data table: leave-one-event-out, and the published split's leakage, side by side.

Reads ``<runs_root>/{loeo,official}/<fold>/<row>/<seed|baseline>/`` (written by
:mod:`floodfusion.crossval`) and writes one Markdown report plus a JSON sidecar.

- **LOEO table.** Every event was tested once, by the fold that held it out: the per-event
  confusion matrices of all folds are pooled into one entry per (row, seed), and the table
  machinery of :mod:`floodfusion.reporting` (basin bootstrap, matched-seed paired gaps,
  the verdict rule) runs on the K events.
- **The leakage gap, on the same chips.** For each event, the *official* model's confusion
  on that event's official test chips is set against the confusion of the LOEO fold that
  held the event out, **re-scored on exactly those chips** (``per_chip_tau_star``). The
  paired bootstrap over events is then a like-for-like measurement of what a within-event
  split adds to a score.
- **Reproducibility** is checked per fold with the same rules as the ablation table.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import ConfusionMatrix, macro_iou, paired_bootstrap_delta
from floodfusion.reporting import (
    MIN_BASINS_FOR_VERDICT,
    ROW_LABELS,
    _check_reproducibility,
    _cm,
    _fmt,
    _gaps,
    _per_basin,
    _row_summary,
    collect_rows,
)
from floodfusion.runs import write_json

__all__ = ["CV_GAPS", "build_cv_table", "merge_folds"]

LOGGER = logging.getLogger(__name__)

#: (row A, row B, question, expected sign of B - A) on real events.
CV_GAPS: tuple[tuple[int, int, str, str], ...] = (
    (1, 2, "What careful engineering adds to naive Otsu", "positive"),
    (2, 3, "What optical adds to the physics baseline", "positive where optical is clear"),
    (2, 4, "What learning adds over the physics baseline (SAR only)", "positive"),
    (4, 9, "What fusion adds over the SAR-only U-Net (R3)", "positive where optical is clear"),
    (11, 9, "The cost of losing optical at inference (9 − 11)", "positive; small is good"),
    (4, 11, "The fusion model run SAR-only vs the SAR-only U-Net", "≈ 0 or positive"),
    (2, 9, "What the machine learning was worth (9 − 2)", "positive"),
)
_ROWS_IN_TABLE = (1, 2, 3, 4, 9, 11)


def _seed_key(entry: Mapping[str, Any]) -> int | None:
    seed = entry["run"].get("seed")
    return int(seed) if isinstance(seed, int) else None


def merge_folds(fold_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    """``{fold: {row: [entries]}}`` -> ``{row: [one merged entry per seed]}``.

    A merged entry's ``per_basin`` is the union of the folds' test events; an event tested
    by two folds is an error (each fold must hold out a different event).
    """
    merged: dict[str, dict[int | None, dict[str, Any]]] = {}
    for fold, rows in sorted(fold_rows.items()):
        for key, entries in rows.items():
            for e in entries:
                slot = merged.setdefault(key, {}).setdefault(_seed_key(e), {
                    "run": {**e["run"], "train_seconds": 0.0, "peak_vram_bytes": 0, "folds": []},
                    "eval": {**e["eval"], "per_basin": {}, "_metrics": []}, "dir": e["dir"].parent.parent,
                })
                clash = set(slot["eval"]["per_basin"]) & set(e["eval"]["per_basin"])
                if clash:
                    raise ValueError(f"row {key}: events {sorted(clash)} tested by more than one fold ({fold})")
                slot["eval"]["per_basin"].update(e["eval"]["per_basin"])
                slot["eval"]["_metrics"].append(e["eval"])
                slot["run"]["folds"].append(fold)
                slot["run"]["train_seconds"] += float(e["run"].get("train_seconds") or 0.0)
                slot["run"]["peak_vram_bytes"] = max(slot["run"]["peak_vram_bytes"], e["run"].get("peak_vram_bytes") or 0)
    out: dict[str, list[dict[str, Any]]] = {}
    for key, by_seed in merged.items():
        out[key] = []
        for slot in by_seed.values():
            metrics = slot["eval"].pop("_metrics")
            cms = {b: _cm(d) for b, d in slot["eval"]["per_basin"].items()}
            slot["eval"]["macro_iou"] = macro_iou(cms)
            for name in ("average_precision", "ece_test_raw", "ece_test_calibrated", "macro_iou_raw_tau_star",
                         "macro_iou_val_selected", "macro_iou_oracle", "pooled_iou"):
                values = [m.get(name) for m in metrics if isinstance(m.get(name), int | float)]
                slot["eval"][name] = float(np.nanmean(values)) if values else float("nan")
            slot["run"]["train_seconds"] = slot["run"]["train_seconds"] / max(len(metrics), 1)
            slot["eval"]["latency"] = None
            slot["eval"]["cloud_free_subset"] = None  # one fold's subset is not the row's
            out[key].append(slot)
    return out


def _collect(protocol_dir: Path, allow_dirty: bool) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[str]]:
    folds, problems = {}, []
    for fold_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
        rows = collect_rows(fold_dir)
        if not rows:
            continue
        problems += _check_reproducibility(rows, allow_dirty)["reproducibility_problems"]
        folds[fold_dir.name] = rows
    return folds, problems


def _cm_of(values: Sequence[int]) -> ConfusionMatrix:
    tp, fp, fn, tn = (int(v) for v in values)
    return ConfusionMatrix(tp, fp, fn, tn)


def _per_event_on_chips(entry: Mapping[str, Any], chips: Sequence[str]) -> dict[str, ConfusionMatrix]:
    per_chip, basin_of = entry["eval"].get("per_chip_tau_star") or {}, entry["eval"].get("per_chip_basin") or {}
    out: dict[str, ConfusionMatrix] = {}
    for chip in chips:
        if chip not in per_chip:
            raise ValueError(f"{entry['dir']}: chip {chip} was not scored (per_chip_tau_star)")
        cm = _cm_of(per_chip[chip])
        basin = basin_of[chip]
        out[basin] = out[basin] + cm if basin in out else cm
    return out


def _leakage(loeo_folds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
             official_folds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
             rows: Sequence[str], n_boot: int, alpha: float) -> tuple[list[str], dict[str, Any]]:
    """Official (within-event) vs LOEO (held-out event) on the SAME test chips, per event."""
    lines = ["| Row | Seed | Events | IoU official split | IoU LOEO, same chips | Δ (official − LOEO) [95 % paired CI] "
             "| Official higher in | Verdict |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    out: dict[str, Any] = {}
    official = official_folds.get("official", {})
    for key in rows:
        for entry in official.get(key, []):
            seed = _seed_key(entry)
            chips_by_event: dict[str, list[str]] = {}
            for chip, basin in (entry["eval"].get("per_chip_basin") or {}).items():
                chips_by_event.setdefault(basin, []).append(chip)
            off = _per_event_on_chips(entry, [c for cs in chips_by_event.values() for c in cs])
            held: dict[str, ConfusionMatrix] = {}
            for event, chips in chips_by_event.items():
                match = [e for e in loeo_folds.get(event, {}).get(key, []) if _seed_key(e) == seed]
                if match:
                    held.update(_per_event_on_chips(match[0], chips))
            common = sorted(set(off) & set(held))
            if len(common) < 2:
                continue
            a, b = {k: held[k] for k in common}, {k: off[k] for k in common}
            delta, lo, hi = paired_bootstrap_delta(a, b, n_boot=n_boot, alpha=alpha, seed=0, statistic="macro")
            wins = sum(b[k].iou > a[k].iou for k in common)
            if len(common) < MIN_BASINS_FOR_VERDICT:
                verdict, resolvable = f"none: K = {len(common)} < {MIN_BASINS_FOR_VERDICT}", None
            else:
                resolvable = lo > 0 or hi < 0
                verdict = "resolvable" if resolvable else "not resolvable (CI contains 0)"
            lines.append(f"| {key} | {seed} | {len(common)} | {_fmt(macro_iou(b))} | {_fmt(macro_iou(a))} | "
                         f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}] | {wins}/{len(common)} | {verdict} |")
            out[f"{key}/seed{seed}"] = {"delta": delta, "ci": [lo, hi], "resolvable": resolvable, "events": common,
                                        "iou_official": macro_iou(b), "iou_loeo_same_chips": macro_iou(a),
                                        "per_event": {k: [b[k].iou, a[k].iou] for k in common}}
    return lines, out


def _bolivia(loeo_folds: Mapping[str, Any], official_folds: Mapping[str, Any], rows: Sequence[str]) -> list[str]:
    lines = ["| Row | Seed | IoU on Bolivia, official model | IoU on Bolivia, LOEO fold |", "| --- | --- | --- | --- |"]
    bolivia = official_folds.get("official_bolivia", {})
    for key in rows:
        for entry in bolivia.get(key, []):
            seed = _seed_key(entry)
            event = next(iter(entry["eval"]["per_basin"]), None)
            loeo = [e for e in loeo_folds.get(event, {}).get(key, []) if _seed_key(e) == seed]
            loeo_iou = _cm(loeo[0]["eval"]["per_basin"][event]).iou if loeo else math.nan
            lines.append(f"| {key} | {seed if seed is not None else '—'} | {_fmt(entry['eval']['macro_iou'])} | {_fmt(loeo_iou)} |")
    return lines


def _conformal(loeo_folds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
               rows: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    """Did the conformal recall guarantee, fitted on the validation EVENT, carry to the held-out event?"""
    lines = ["| Row | Seed | α | Events where mean chip FNR ≤ α | Mean chip FNR over events | Worst event |",
             "| --- | --- | --- | --- | --- | --- |"]
    out: dict[str, Any] = {}
    for key in rows:
        by_seed: dict[int | None, dict[str, float]] = {}
        alpha = math.nan
        for _, fold_rows in sorted(loeo_folds.items()):
            for e in fold_rows.get(key, []):
                c = e["eval"].get("conformal")
                if not c:
                    continue
                alpha = c["alpha"]
                by_seed.setdefault(_seed_key(e), {}).update(c.get("test_mean_chip_fnr_by_basin", {}))
        for seed, per_event in by_seed.items():
            if not per_event:
                continue
            held = sum(v <= alpha for v in per_event.values())
            worst = max(per_event, key=per_event.get)
            lines.append(f"| {key} | {seed} | {alpha:g} | {held}/{len(per_event)} | "
                         f"{np.mean(list(per_event.values())):.3f} | {worst} ({per_event[worst]:.3f}) |")
            out[f"{key}/seed{seed}"] = {"alpha": alpha, "per_event_mean_chip_fnr": per_event, "events_held": held}
    return lines, out


def build_cv_table(runs_root: Path | str, *, out: Path | str | None = None, allow_dirty: bool = False,
                   n_boot: int = 10_000, alpha: float = 0.05) -> str:
    runs_root = Path(runs_root)
    loeo_folds, problems = _collect(runs_root / "loeo", allow_dirty)
    if not loeo_folds:
        raise FileNotFoundError(f"no evaluated LOEO folds under {runs_root / 'loeo'}")
    official_folds, more = (_collect(runs_root / "official", allow_dirty) if (runs_root / "official").is_dir()
                            else ({}, []))
    problems += more
    rows = merge_folds(loeo_folds)
    summaries = {k: _row_summary(k, v, n_boot, alpha) for k, v in rows.items()}
    gap_lines, gaps = _gaps(summaries, rows, n_boot, alpha, gaps=CV_GAPS)
    leak_lines, leakage = _leakage(loeo_folds, official_folds, ["4", "9", "11"], n_boot, alpha)
    conformal_lines, conformal = _conformal(loeo_folds, ["4", "9", "11"])
    events = sorted({b for s in summaries.values() for b in s["per_basin"]})
    main = ["| # | Configuration | IoU at τ* [95 % CI over events] | F1 | Seeds | Events |", "| --- | --- | --- | --- | --- | --- |"]
    for row in _ROWS_IN_TABLE:
        s = summaries.get(str(row))
        if s is None:
            continue
        mark = "†" if s["n_basins"] < MIN_BASINS_FOR_VERDICT else ""
        main.append(f"| {row} | {ROW_LABELS[row][0]} | {_fmt(s['iou'])} [{_fmt(s['ci'][0], 2)}, {_fmt(s['ci'][1], 2)}]{mark} "
                    f"| {_fmt(s['f1'])} | {', '.join(map(str, s['seeds'])) or '—'} | {s['n_basins']} |")
    text = "\n".join([
        "# Sen1Floods11, re-ingested — leave-one-event-out and the published split",
        "",
        "Real data: Sen1Floods11 hand labels (Bonafilia et al., 2020) with every input re-fetched from "
        "Planetary Computer by `floodfusion.data.sen1floods11` (same-orbit pre-event RTC, S2 L2A, GLO-30, JRC, "
        "WorldCover; optical and labels moved onto the SAR geometry). Flood = labelled water minus JRC "
        "permanent water. IoU at τ* = 0.25 after calibration on each fold's validation event.",
        "",
        f"**Protocol: leave-one-event-out.** {len(events)} folds; each event is tested once by a model that never "
        "saw it; validation = the next event (sorted, cyclic), fixed before any run. **[CI]** = 95 % percentile "
        "bootstrap over test events; with K ≥ 8 events the gaps below carry a verdict.",
        "",
        *main, "",
        "## Paired gaps over events (matched seeds)", "", *gap_lines, "",
        "## The published split's leakage, measured on the same chips", "",
        "The dataset's official test chips come from the **same ten events** as its training chips. Each event's "
        "official test chips are scored twice: by the model trained on the official split, and by the LOEO model "
        "that never saw the event. Positive Δ = what the within-event split adds to the score.", "",
        *(leak_lines if len(leak_lines) > 2 else ["(official protocol not run)"]), "",
        "## Bolivia — the dataset's own held-out event", "", *_bolivia(loeo_folds, official_folds, ["4", "9", "11"]), "",
        "## Conformal recall control across events", "",
        "Each fold fits the conformal threshold on its validation event's chips (risk = per-chip false-negative "
        "rate, guarantee E[risk] ≤ α for chips exchangeable with those). The held-out event is a different flood: "
        "this measures how far the guarantee carries.", "",
        *(conformal_lines if len(conformal_lines) > 2 else ["(no conformal results recorded)"]), "",
        "## Per-event IoU at τ* (LOEO)", "", *_per_basin(summaries), "",
        *(["## Reproducibility problems", "", *[f"- {p}" for p in problems[:30]], ""] if problems else []),
    ])
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        write_json(out.with_suffix(".json"), {
            "rows": {k: {"iou": s["iou"], "ci": list(s["ci"]), "per_event": {b: m.iou for b, m in s["per_basin"].items()},
                         "seeds": s["seeds"]} for k, s in summaries.items()},
            "gaps": gaps, "leakage": leakage, "conformal": conformal, "events": events,
            "reproducibility_problems": problems,
        })
    LOGGER.info("wrote %s", out)
    return text

