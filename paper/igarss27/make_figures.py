"""Figures of the IGARSS 2027 paper, drawn from committed reports only; every plotted number is printed and saved.

Run from the repository root::

    uv run --frozen python paper/igarss27/make_figures.py

Figure 1 (exploratory, chosen after both inputs were known): for each seen activation in both reports, the v1.2 model's
IoU at tau* against the agreement of two Copernicus EMS delineations of the same area at most 24 h apart.
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


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = fig_ceiling(OUT)
    print(json.dumps({k: v for k, v in summary.items() if k != "activations"}))  # noqa: T201 - the numbers plotted


if __name__ == "__main__":
    main()
