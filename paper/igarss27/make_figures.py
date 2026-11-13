"""Figures of the IGARSS 2027 paper, drawn from committed reports only; every plotted number is printed and saved.

Run from the repository root::

    uv run --frozen python paper/igarss27/make_figures.py

Figure 1 (exploratory, chosen after both inputs were known): for each seen activation in both reports, the v1.2 model's
IoU at tau* against the agreement of two Copernicus EMS delineations of the same area at most 24 h apart.

Figure 2: the ladder's per-activation scores, R1 (zero-shot) against R3 (fine-tuned on Copernicus labels), from
reports/v13_ladder.json; the count above the diagonal is checked against the report's own verdict. Its JSON also holds
the per-activation change's median and extremes and the per-fold means, which the long paper (paper_v13.tex) cites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "figures"
DPI = 300
COLUMN_IN = 3.45            # IEEE two-column width
BLUE, GREY = "#0072B2", "0.45"


def _style() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def ceiling_pairs(reports: Path = REPORTS) -> dict[str, tuple[float, float]]:
    """``{activation: (Copernicus self-agreement, v1.2 IoU at tau*)}`` over the activations in both reports."""
    table = json.loads((reports / "v12_cems_rebuilt.json").read_text(encoding="utf-8"))["tables"][0]
    if table["unit"] != "activation" or table["gfm_common"]:
        raise ValueError("the first table of v12_cems_rebuilt.json is no longer the per-activation, all-units table")
    model = {k: v["iou_tau"] for k, v in table["models"]["ours"]["per_unit"].items()}
    ceiling = json.loads((reports / "reference_ceiling.json").read_text(encoding="utf-8"))["summary"]["all"]
    pairs = {k: (float(ceiling["per_activation"][k]), float(model[k]))
             for k in sorted(set(model) & set(ceiling["per_activation"]))
             if ceiling["per_activation"][k] is not None and model[k] is not None}
    return {k: v for k, v in pairs.items() if np.isfinite(v).all()}


def ceiling_stats(pairs: dict[str, tuple[float, float]]) -> dict[str, Any]:
    x, y = np.array(list(pairs.values())).T
    rho, p = stats.spearmanr(x, y)
    return {"n": len(pairs), "spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4),
            "model_below_ceiling": int((y < x).sum()), "macro_ceiling": round(float(x.mean()), 3),
            "macro_model": round(float(y.mean()), 3), "activations": sorted(pairs)}


def fig_ceiling(out: Path) -> dict[str, Any]:
    pairs = ceiling_pairs()
    summary = ceiling_stats(pairs)
    x, y = np.array(list(pairs.values())).T
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--", zorder=1)   # equal agreement (in the caption)
    ax.scatter(x, y, s=14, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="IoU between two Copernicus EMS maps, $\\leq$ 24 h apart",
           ylabel="v1.2 IoU at $\\tau^*$ against Copernicus EMS")
    ax.text(0.03, 0.97, f"{summary['n']} activations\nSpearman $\\rho$ = {summary['spearman_rho']:.2f}\n"
            f"model below: {summary['model_below_ceiling']} of {summary['n']}", ha="left", va="top", fontsize=6.5)
    ax.set_aspect("equal")
    fig.savefig(out / "ceiling_scatter.pdf", bbox_inches="tight")
    fig.savefig(out / "ceiling_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    (out / "ceiling_scatter.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def ladder_pairs(reports: Path = REPORTS) -> tuple[dict[str, tuple[float, float]], dict[str, Any], dict[str, str]]:
    """``{activation: (R1 IoU, R3 IoU)}`` from the committed ladder report, its R3 verdict, and each activation's fold."""
    ladder = json.loads((reports / "v13_ladder.json").read_text(encoding="utf-8"))
    verdict = ladder["verdicts"]["r3_over_base"]
    if verdict["base"] != "r1":
        raise ValueError("R3 is no longer judged against R1 in v13_ladder.json; the figure would mislabel its axis")
    pairs = {k: (float(ladder["r1"][k]), float(ladder["r3"][k])) for k in sorted(ladder["r1"])}
    if len(pairs) != verdict["n"]:
        raise ValueError(f"{len(pairs)} activations in r1, but the verdict counts {verdict['n']}")
    return pairs, verdict, {k: ladder["fold_of"][k] for k in pairs}


def _fold_summary(pairs: dict[str, tuple[float, float]], fold_of: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Per fold (descriptive, for the long paper): its activations' macro R1 and R3, and how many R3 improves."""
    folds: dict[str, dict[str, Any]] = {}
    for fold in sorted(set(fold_of.values())):
        r1, r3 = np.array([pairs[k] for k in pairs if fold_of[k] == fold]).T
        folds[fold] = {"n": len(r1), "macro_r1": round(float(r1.mean()), 3), "macro_r3": round(float(r3.mean()), 3),
                       "r3_better": int((r3 > r1).sum())}
    return folds


def ladder_stats(pairs: dict[str, tuple[float, float]], verdict: dict[str, Any],
                 fold_of: dict[str, str]) -> dict[str, Any]:
    x, y = np.array(list(pairs.values())).T
    better = int((y > x).sum())
    if better != verdict["better"]:   # the plot must show the report's own count
        raise ValueError(f"{better} activations above the diagonal, but the report says {verdict['better']}")
    change = {k: r3 - r1 for k, (r1, r3) in pairs.items()}
    best, worst = max(change, key=change.__getitem__), min(change, key=change.__getitem__)
    return {"n": len(pairs), "r3_better": better, "macro_r1": round(float(x.mean()), 3),
            "macro_r3": round(float(y.mean()), 3), "gain": round(float(y.mean() - x.mean()), 3),
            "adopted": bool(verdict["adopted"]),
            "change_median": round(float(np.median(list(change.values()))), 3),
            "change_max": {"activation": best, "change": round(change[best], 3)},
            "change_min": {"activation": worst, "change": round(change[worst], 3)},
            "folds": _fold_summary(pairs, fold_of), "activations": sorted(pairs)}


def fig_ladder(out: Path) -> dict[str, Any]:
    pairs, verdict, fold_of = ladder_pairs()
    summary = ladder_stats(pairs, verdict, fold_of)
    x, y = np.array(list(pairs.values())).T
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--", zorder=1)   # no change (in the caption)
    ax.scatter(x, y, s=14, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="R1, zero-shot: IoU at $\\tau^*$ against Copernicus EMS",
           ylabel="R3, fine-tuned on CEMS labels")
    ax.text(0.03, 0.97, f"{summary['n']} activations\nmacro {summary['macro_r1']:.3f} $\\to$ {summary['macro_r3']:.3f}\n"
            f"R3 better: {summary['r3_better']} of {summary['n']}", ha="left", va="top", fontsize=6.5)
    ax.set_aspect("equal")
    fig.savefig(out / "ladder_r3.pdf", bbox_inches="tight")
    fig.savefig(out / "ladder_r3.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    (out / "ladder_r3.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for summary in (fig_ceiling(OUT), fig_ladder(OUT)):
        print(json.dumps({k: v for k, v in summary.items() if k != "activations"}))  # noqa: T201 - the numbers plotted


if __name__ == "__main__":
    main()
