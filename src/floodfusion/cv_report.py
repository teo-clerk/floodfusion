r"""The real-data table: leave-one-event-out, and the published split's protocol gap, side by side.

Reads ``<runs_root>/{loeo,official}/<fold>/<row>/<seed|baseline>/`` (written by
:mod:`floodfusion.crossval`) and writes one Markdown report plus a JSON sidecar.

- **LOEO table.** Every event was tested once, by the fold that held it out: the per-event
  confusion matrices of all folds are pooled into one entry per (row, seed), and the gap
  machinery of :mod:`floodfusion.reporting` runs on the K events (exact sign-flip tests,
  Holm-adjusted). AP and the oracle-threshold IoU are shown beside IoU at τ*, so ranking
  quality and threshold transfer can be told apart.
- **The protocol gap, on the same chips.** For each event, the official-split model's
  confusion on that event's official test chips is set against the LOEO fold that held the
  event out, **re-scored on exactly those chips** — at τ* (each model's own calibration),
  at each model's best threshold on the event, and threshold-free (AP). The difference
  combines leakage (the official model saw the event) with the protocols' other
  differences (training-set size, validation set, calibration); it is reported as such.
- **Conformal recall control** per held-out event: feasibility, calibration size, the
  test per-chip miss rate, and the price paid (precision, IoU at the conformal threshold).
- **Reproducibility** is checked per fold with the same rules as the ablation table.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.metrics import (
    ConfusionMatrix,
    average_precision,
    holm_adjust,
    macro_iou,
    precision_recall_curve,
    sign_flip_p_value,
    sum_confusions,
)
from floodfusion.reporting import (
    MIN_BASINS_FOR_VERDICT,
    ROW_LABELS,
    check_reproducibility,
    cm_from_dict,
    collect_rows,
    fmt,
    paired_gaps,
    per_basin_table,
    row_summary,
)
from floodfusion.runs import write_json

__all__ = ["CAVEATS", "CV_GAPS", "build_cv_table", "merge_folds", "protocol_gap"]

LOGGER = logging.getLogger(__name__)

TERRAMIND = "cmp_terramind_small"
#: (row A, row B, question, expected sign of B - A) on real events.
CV_GAPS: tuple[tuple[int | str, int | str, str, str], ...] = (
    (1, 2, "What careful engineering adds to naive Otsu", "positive"),
    (2, 3, "What optical adds to the physics baseline †", "positive where optical is clear"),
    (2, 4, "What learning adds over the physics baseline (SAR only)", "positive"),
    (4, 9, "What fusion adds over the SAR-only U-Net (R3) †", "positive where optical is clear"),
    (11, 9, "The cost of losing optical at inference (9 − 11) †", "positive; small is good"),
    (4, 11, "The fusion model run SAR-only vs the SAR-only U-Net", "≈ 0 or positive"),
    (2, 9, "What the machine learning was worth (9 − 2)", "positive"),
    (4, TERRAMIND, "TerraMind (frozen; post-event S1 + S2) vs the SAR-only U-Net ‡", "open"),
    (TERRAMIND, 9, "What the task-specific fusion model adds over the foundation model ‡",
     "≈ 0 on Sen1Floods11 (PANGAEA: supervised U-Nets match GFMs there)"),
)
_ROWS_IN_TABLE: tuple[int | str, ...] = (1, 2, 3, 4, 9, 11, TERRAMIND)
_LABELS = {TERRAMIND: "TerraMind-1.0-small, frozen + UNet decoder (single date: post S1 + S2) ‡"}
_GAP_ROWS = ("4", "9", "11", TERRAMIND)

#: Stated wherever the numbers they qualify appear.
CAVEATS = (
    "† **Optical is the annotators' source image.** Sen1Floods11 labels were drawn mostly on the Sentinel-2 "
    "image of the same date, and that image is re-fetched as the optical input. Gaps involving optical "
    "therefore measure agreement with the annotators' source as well as the value of optical; the EMSR664 "
    "case study (independent CEMS delineations) is the independent check.",
    "‡ **TerraMind** saw global Sentinel-1/2 data in pre-training (\"never saw the event\" holds for its "
    "decoder only); its band subset and necks follow IBM's recipe, developed on Sen1Floods11's official "
    "splits; its schedule is row 9's, tuned for neither model; its SAR input is our z-score adapted by four "
    "affine parameters; clouded pixels reach it as its pre-training mean (it has no per-pixel gate).",
    "**Labels mark water on a date, not a change.** In 7 of 10 events the flood-labelled pixels show no "
    "backscatter decrease against a same-orbit reference up to 60 days earlier (see the label diagnostic): "
    "change-based rules (rows 2, 3) are structurally penalised, amplitude-based ones are not.",
    "**Validation is one event per fold** (the next, fixed in advance): early stopping, the prior, the "
    "calibration and the baseline offset come from it, so part of a fold's score is threshold transfer — "
    "compare IoU at τ* with the oracle-threshold IoU and AP.",
    "**Development runs** scored one fold (Bolivia; rows 1-4) before the protocol was frozen; every later "
    "change came from code review, none from those numbers.",
)


def _seed_key(entry: Mapping[str, Any]) -> int | None:
    seed = entry["run"].get("seed")
    return int(seed) if isinstance(seed, int) else None


def merge_folds(fold_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    """``{fold: {row: [entries]}}`` -> ``{row: [one merged entry per seed]}``.

    A merged entry's ``per_basin`` is the union of the folds' test events; an event tested
    by two folds is an error (each fold must hold out a different event). Per-fold scalar
    metrics (AP, oracle IoU, ECE) are averaged: one fold is one event, so that is their macro.
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
            slot["eval"]["macro_iou"] = macro_iou({b: cm_from_dict(d) for b, d in slot["eval"]["per_basin"].items()})
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
    if not protocol_dir.is_dir():
        return folds, problems
    for fold_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
        rows = collect_rows(fold_dir)
        if rows:
            problems += check_reproducibility(rows, allow_dirty)["reproducibility_problems"]
            folds[fold_dir.name] = rows
    return folds, problems


# ---------------------------------------------------------------------------
# Per-event scores from per-chip confusion (any subset of chips, any grid threshold)
# ---------------------------------------------------------------------------
def _grid_by_event(entry: Mapping[str, Any], chips: Sequence[str]) -> dict[str, list[ConfusionMatrix]]:
    """Per event: one confusion per grid threshold, summed over ``chips``."""
    grid = entry["eval"].get("per_chip_grid") or {}
    cms, basin_of = grid.get("cms") or {}, entry["eval"].get("per_chip_basin") or {}
    out: dict[str, list[ConfusionMatrix]] = {}
    for chip in chips:
        if chip not in cms:
            raise ValueError(f"{entry['dir']}: chip {chip} was not scored on the threshold grid")
        row = [ConfusionMatrix(*map(int, v)) for v in cms[chip]]
        event = basin_of[chip]
        out[event] = row if event not in out else [a + b for a, b in zip(out[event], row)]
    return out


def _tau_by_event(entry: Mapping[str, Any], chips: Sequence[str]) -> dict[str, ConfusionMatrix]:
    per_chip, basin_of = entry["eval"].get("per_chip_tau_star") or {}, entry["eval"].get("per_chip_basin") or {}
    out: dict[str, ConfusionMatrix] = {}
    for chip in chips:
        if chip not in per_chip:
            raise ValueError(f"{entry['dir']}: chip {chip} was not scored (per_chip_tau_star)")
        cm, event = ConfusionMatrix(*map(int, per_chip[chip])), basin_of[chip]
        out[event] = out[event] + cm if event in out else cm
    return out


def _event_scores(tau: Mapping[str, ConfusionMatrix], grid: Mapping[str, Sequence[ConfusionMatrix]]) -> dict[str, dict[str, float]]:
    scores = {}
    for event, cm in tau.items():
        curve = grid[event]
        ious = [c.iou for c in curve if not math.isnan(c.iou)]
        try:
            ap = average_precision(*precision_recall_curve(tuple(curve)))
        except ValueError:
            ap = float("nan")
        scores[event] = {"iou_tau": cm.iou, "iou_oracle": max(ious) if ious else float("nan"), "ap": ap}
    return scores


def protocol_gap(official: Mapping[str, Any], held_out: Mapping[str, Mapping[str, Any]],
                 *, exclude: Sequence[str] = ()) -> dict[str, Any] | None:
    """Official-split model vs the LOEO models, per event, on the official test chips of each event.

    ``held_out`` maps an event to the LOEO entry (same row and seed) of the fold that held it out.
    Returns per-event scores of both, and per measure (IoU at τ*, oracle IoU, AP) the mean
    difference (official − LOEO), its per-event signs and an exact sign-flip p-value.
    """
    chips_by_event: dict[str, list[str]] = {}
    for chip, event in (official["eval"].get("per_chip_basin") or {}).items():
        if event not in exclude:
            chips_by_event.setdefault(event, []).append(chip)
    events = sorted(e for e in chips_by_event if e in held_out)
    if len(events) < 2:
        return None
    chips = [c for e in events for c in chips_by_event[e]]
    off = _event_scores(_tau_by_event(official, chips), _grid_by_event(official, chips))
    loeo = {}
    for event in events:
        entry = held_out[event]
        mine = chips_by_event[event]
        loeo.update(_event_scores(_tau_by_event(entry, mine), _grid_by_event(entry, mine)))
    result: dict[str, Any] = {"events": events, "per_event": {e: {"official": off[e], "loeo": loeo[e]} for e in events}}
    for measure in ("iou_tau", "iou_oracle", "ap"):
        deltas = [off[e][measure] - loeo[e][measure] for e in events]
        finite = [d for d in deltas if math.isfinite(d)]
        result[measure] = {"official": float(np.nanmean([off[e][measure] for e in events])),
                           "loeo": float(np.nanmean([loeo[e][measure] for e in events])),
                           "delta": float(np.mean(finite)) if finite else float("nan"),
                           "official_higher": sum(d > 0 for d in finite), "n": len(finite),
                           "p_value": sign_flip_p_value(finite)}
    return result


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def _main_table(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| # | Configuration | IoU at τ* [95 % CI over events] | IoU, oracle τ | AP | F1 | Seeds | Events |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in _ROWS_IN_TABLE:
        s = summaries.get(str(row))
        if s is None:
            continue
        mark = "§" if s["n_basins"] < MIN_BASINS_FOR_VERDICT else ""
        label = _LABELS.get(row) or ROW_LABELS[row][0]
        lines.append(f"| {row if isinstance(row, int) else 'TM'} | {label} | {fmt(s['iou'])} "
                     f"[{fmt(s['ci'][0], 2)}, {fmt(s['ci'][1], 2)}]{mark} | {fmt(s['oracle'])} | {fmt(s['ap'])} | "
                     f"{fmt(s['f1'])} | {', '.join(map(str, s['seeds'])) or '—'} | {s['n_basins']} |")
    return lines


def _protocol_section(loeo_folds: Mapping[str, Any], official_folds: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    lines = ["| Row | Seed | Events | Measure | Official split | LOEO, same chips | Δ (official − LOEO) | Official higher in | p (sign-flip) | Holm p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    official = official_folds.get("official", {})
    found: dict[str, Any] = {}
    for key in _GAP_ROWS:
        for entry in official.get(key, []):
            seed = _seed_key(entry)
            held = {}
            for event, fold_rows in loeo_folds.items():
                match = [e for e in fold_rows.get(key, []) if _seed_key(e) == seed]
                if match:
                    held[event] = match[0]
            gap = protocol_gap(entry, held, exclude=("S1F11_Bolivia",))
            if gap is not None:
                found[f"{key}/seed{seed}"] = gap
    holm = holm_adjust({f"{k}/{m}": g[m]["p_value"] for k, g in found.items() for m in ("iou_tau", "iou_oracle", "ap")
                        if g[m]["n"] >= MIN_BASINS_FOR_VERDICT})
    for name, gap in found.items():
        key, seed = name.split("/seed")
        for measure, label in (("iou_tau", "IoU at τ*"), ("iou_oracle", "IoU, oracle τ"), ("ap", "AP")):
            m = gap[measure]
            adj = holm.get(f"{name}/{measure}")
            lines.append(f"| {key} | {seed} | {len(gap['events'])} | {label} | {fmt(m['official'])} | {fmt(m['loeo'])} | "
                         f"{m['delta']:+.3f} | {m['official_higher']}/{m['n']} | {fmt(m['p_value'])} | "
                         f"{fmt(adj) if adj is not None else '—'} |")
            m["p_holm"] = adj
    return lines, found


def _bolivia_section(loeo_folds: Mapping[str, Any], official_folds: Mapping[str, Any]) -> list[str]:
    lines = ["| Row | Seed | IoU on Bolivia, official-split model | IoU on Bolivia, LOEO fold |", "| --- | --- | --- | --- |"]
    official, event = official_folds.get("official", {}), "S1F11_Bolivia"
    for key in _GAP_ROWS:
        for entry in official.get(key, []):
            if event not in entry["eval"]["per_basin"]:
                continue
            seed = _seed_key(entry)
            loeo = [e for e in loeo_folds.get(event, {}).get(key, []) if _seed_key(e) == seed]
            loeo_iou = cm_from_dict(loeo[0]["eval"]["per_basin"][event]).iou if loeo else math.nan
            lines.append(f"| {key} | {seed if seed is not None else '—'} | "
                         f"{fmt(cm_from_dict(entry['eval']['per_basin'][event]).iou)} | {fmt(loeo_iou)} |")
    return lines


def _conformal_section(loeo_folds: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    lines = ["| Row | Seed | α | Feasible folds | Events with mean chip FNR ≤ α (feasible folds) | Mean chip FNR | "
             "Worst event | Precision at τ_conf | IoU at τ_conf |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out: dict[str, Any] = {}
    for key in _GAP_ROWS:
        per_seed: dict[int | None, list[dict[str, Any]]] = {}
        for fold_rows in loeo_folds.values():
            for e in fold_rows.get(key, []):
                if e["eval"].get("conformal"):
                    per_seed.setdefault(_seed_key(e), []).append(e["eval"]["conformal"])
        for seed, folds in per_seed.items():
            feasible = [c for c in folds if c.get("feasible")]
            fnr = {b: v for c in feasible for b, v in c.get("test_mean_chip_fnr_by_basin", {}).items()}
            alpha = folds[0]["alpha"]
            held = sum(v <= alpha for v in fnr.values())
            worst = max(fnr, key=fnr.get) if fnr else None
            lines.append(f"| {key} | {seed} | {alpha:g} | {len(feasible)}/{len(folds)} | {held}/{len(fnr)} | "
                         f"{fmt(float(np.mean(list(fnr.values()))) if fnr else math.nan)} | "
                         f"{f'{worst} ({fnr[worst]:.3f})' if worst else '—'} | "
                         f"{fmt(float(np.nanmean([c.get('test_precision', math.nan) for c in feasible])) if feasible else math.nan)} | "
                         f"{fmt(float(np.nanmean([c.get('test_iou', math.nan) for c in feasible])) if feasible else math.nan)} |")
            out[f"{key}/seed{seed}"] = {"alpha": alpha, "feasible": len(feasible), "folds": len(folds),
                                        "per_event_mean_chip_fnr": fnr, "events_held": held,
                                        "n_calibration_chips": [c.get("n") for c in folds]}
    return lines, out


def _attribution_section(loeo_folds: Mapping[str, Any]) -> list[str]:
    """Row 2 per event with each rule component disabled — the change term above all."""
    variants: dict[str, dict[str, ConfusionMatrix]] = {}
    fallbacks, scenes = 0, 0
    for fold_rows in loeo_folds.values():
        for e in fold_rows.get("2", []):
            for name, per_basin in (e["eval"].get("attribution_per_basin") or {}).items():
                variants.setdefault(name, {}).update({b: cm_from_dict(d) for b, d in per_basin.items()})
            for info in (e["eval"].get("scenes") or {}).values():
                scenes += 1
                fallbacks += info.get("fallback") == "global_otsu"
    if not variants:
        return []
    lines = [f"The tile-adaptive threshold fell back to global Otsu on {fallbacks} of {scenes} scenes "
             "(a 5 km chip offers few tiles with a bimodal histogram).", "",
             "| Row 2 variant | Macro IoU over events | Pooled IoU |", "| --- | --- | --- |"]
    lines += [f"| {name} | {fmt(macro_iou(pb))} | {fmt(sum_confusions(list(pb.values())).iou)} |"
              for name, pb in sorted(variants.items())]
    return lines


def build_cv_table(runs_root: Path | str, *, out: Path | str | None = None, allow_dirty: bool = False,
                   n_boot: int = 10_000, alpha: float = 0.05) -> str:
    runs_root = Path(runs_root)
    loeo_folds, problems = _collect(runs_root / "loeo", allow_dirty)
    if not loeo_folds:
        raise FileNotFoundError(f"no evaluated LOEO folds under {runs_root / 'loeo'}")
    official_folds, more = _collect(runs_root / "official", allow_dirty)
    problems += more
    rows = merge_folds(loeo_folds)
    summaries = {k: row_summary(k, v, n_boot, alpha) for k, v in rows.items()}
    gap_lines, gaps = paired_gaps(summaries, rows, n_boot, alpha, gaps=CV_GAPS)
    protocol_lines, protocol = _protocol_section(loeo_folds, official_folds)
    conformal_lines, conformal = _conformal_section(loeo_folds)
    attribution = _attribution_section(loeo_folds)
    events = sorted({b for s in summaries.values() for b in s["per_basin"]})
    text = "\n".join([
        "# Sen1Floods11, re-ingested — leave-one-event-out and the published split",
        "",
        "Real data: Sen1Floods11 hand labels (Bonafilia et al., 2020) with every input re-fetched from Planetary "
        "Computer by `floodfusion.data.sen1floods11` (same-orbit pre-event RTC, S2 L2A, GLO-30, JRC, WorldCover; "
        "optical and labels moved onto the SAR geometry). Flood = labelled water minus JRC permanent water. IoU at "
        "τ* = 0.25 after calibration on each fold's validation event.",
        "",
        f"**Protocol: leave-one-event-out.** {len(events)} folds (Paraguay dropped: its acquisition is not in the "
        "archive); each event is tested once by a model that never saw it; validation = the next event (sorted, "
        "cyclic), fixed before any run. **[CI]** = 95 % percentile bootstrap over events (descriptive); gap verdicts "
        "are exact sign-flip permutation tests over events, Holm-adjusted across the table.",
        "",
        *_main_table(summaries), "",
        f"§ fewer than {MIN_BASINS_FOR_VERDICT} events: the interval is indicative only.", "",
        "## Caveats", "", *[f"- {c}" for c in CAVEATS], "",
        "## Paired gaps over events (matched seeds)", "", *gap_lines, "",
        "## The published split's protocol gap, measured on the same chips", "",
        "The dataset's official test chips come from the **same events** as its training chips. Each event's official "
        "test chips are scored twice: by the model trained on the official split, and by the LOEO model that never "
        "saw the event. Δ combines leakage with the protocols' other differences (training-set size, validation set, "
        "calibration): at τ* it includes each model's calibration; at the oracle threshold and in AP it does not.", "",
        *(protocol_lines if len(protocol_lines) > 2 else ["(official protocol not run)"]), "",
        "## Bolivia — the dataset's own held-out event", "", *_bolivia_section(loeo_folds, official_folds), "",
        "## Conformal recall control across events", "",
        "Each fold fits the conformal threshold on its validation event's chips (≥ 100 flood pixels; risk = "
        "per-chip false-negative rate; guarantee E[risk] ≤ α for chips exchangeable with those — windows of one "
        "source chip are not independent, and the same chips also selected the checkpoint). The held-out event "
        "is a different flood: this measures how far the guarantee carries.", "",
        *(conformal_lines if len(conformal_lines) > 2 else ["(no conformal results recorded)"]), "",
        *(["## Physics baseline: what each rule component costs on these labels", "", *attribution, ""]
          if attribution else []),
        "## Per-event IoU at τ* (LOEO)", "", *per_basin_table(summaries), "",
        *(["## Reproducibility problems", "", *[f"- {p}" for p in problems[:30]], ""] if problems else []),
    ])
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        write_json(out.with_suffix(".json"), {
            "rows": {k: {"iou": s["iou"], "ci": list(s["ci"]), "ap": s["ap"], "oracle": s["oracle"],
                         "per_event": {b: m.iou for b, m in s["per_basin"].items()}, "seeds": s["seeds"]}
                     for k, s in summaries.items()},
            "gaps": gaps, "protocol_gap": protocol, "conformal": conformal, "events": events,
            "reproducibility_problems": problems,
        })
    LOGGER.info("wrote %s", out)
    return text
