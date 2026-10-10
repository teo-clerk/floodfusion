"""The figures that earn their place in the report (``docs/09`` section 8.2).

- :func:`plot_pr_curve` — figure 3: PR curve with the operating point
  :math:`\\tau^\\star = 1/(1+r)` marked and the cost ratio stated.
- :func:`plot_reliability` — figure 6: reliability diagram before and after calibration
  (quantile bins).
- :func:`plot_basin_dots` — figure 4: per-basin IoU with bootstrap CIs per row; it shows
  the spread the mean hides.

Headless (``Agg``); every function writes a PNG and returns its path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["plot_basin_dots", "plot_pr_curve", "plot_reliability"]


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr_curve(
    precisions: Sequence[float],
    recalls: Sequence[float],
    operating_point: tuple[float, float],
    out: Path,
    *,
    title: str,
    cost_ratio: float,
    tau: float,
    ap: float,
) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=150)
    ax.step(recalls, precisions, where="post", lw=1.6, label=f"AP = {ap:.3f}")
    ax.plot(operating_point[1], operating_point[0], "o", ms=8, color="crimson",
            label=f"τ* = 1/(1+r) = {tau:.2f}  (r = c_FN/c_FP = {cost_ratio:g})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall", ylabel="precision", title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_reliability(before: Mapping[str, Any], after: Mapping[str, Any], out: Path, *, title: str,
                     ece_before: float, ece_after: float) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for diag, label, marker in ((before, f"raw (ECE {ece_before:.3f})", "s"),
                                (after, f"prior + temperature (ECE {ece_after:.3f})", "o")):
        conf, freq = np.asarray(diag["confidence"]), np.asarray(diag["frequency"])
        keep = np.isfinite(conf) & np.isfinite(freq)
        ax.plot(conf[keep], freq[keep], marker=marker, ms=4, lw=1.2, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability (quantile bins)",
           ylabel="observed flood frequency", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_basin_dots(rows: Sequence[Mapping[str, Any]], out: Path, *, title: str) -> Path:
    """``rows``: dicts with ``label``, ``per_basin`` ({basin: IoU}), ``iou``, ``ci`` (lo, hi)."""
    plt = _pyplot()
    ordered = sorted(rows, key=lambda r: (np.nan_to_num(r["iou"], nan=-1.0)))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ordered) + 1.4), dpi=150)
    for y, row in enumerate(ordered):
        lo, hi = row["ci"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color="0.55", lw=3, solid_capstyle="butt", zorder=1)
        values = [v for v in row["per_basin"].values() if v is not None and np.isfinite(v)]
        ax.scatter(values, [y] * len(values), s=14, color="tab:blue", alpha=0.7, zorder=2)
        if np.isfinite(row["iou"]):
            ax.scatter([row["iou"]], [y], s=42, marker="D", color="crimson", zorder=3)
    ax.set_yticks(range(len(ordered)), [r["label"] for r in ordered], fontsize=7)
    ax.set(xlim=(0, 1), xlabel="IoU at τ* (dots: test basins; diamond: macro mean; bar: 95 % basin bootstrap)",
           title=title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
