"""Figures for the v1.2 paper (``paper/paper_v12.tex``), read from the committed reports; nothing is re-scored.

    uv run --frozen python paper/make_figures.py     # writes paper/figures/*.png

- ``cv_vs_cems.png``: each model's cross-validation score beside its score on the 38 Copernicus EMS activations.
- ``per_activation.png``: v1.2's model against the fair TerraMind, v1.1's best and Copernicus GFM, one point per
  activation (the units of ``reports/v12_cems_seen_tests.md``).
- ``emsr657_<layer>.png``: the Arena's quicklooks of one flood (EMSR657 AOI01), converted from WebP.

The CV values are parsed from the report tables (``reports/*.md``) and the CEMS values are read from
``reports/v12_cems_rebuilt.json``; the script prints every plotted number so it can be checked against the text.

Contains modified Copernicus Sentinel data and modified Copernicus Emergency Management Service information
(2023–2026), © European Union.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "figures"
DPI = 300
COLUMN_IN, PAGE_IN = 3.35, 6.9
# Okabe–Ito, as in the Arena: one colour per training set.
TRAINING_COLOURS = {"Sen1Floods11": "#E69F00", "Kuro Siwo": "#0072B2", "Kuro Siwo + GEOID-Flood": "#CC79A7"}
QUICKLOOK_PAIR = "EMSR657_AOI01"
QUICKLOOK_LAYERS = ("radar", "cems", "ks_r9_flood", "gfm")
QUICKLOOK_SIDE = 512


@dataclass(frozen=True)
class Point:
    """One model: its CV score (from a report table) and its CEMS score (from the rebuilt-stack scores)."""

    label: str
    training: str
    cv_protocol: str
    cv_iou: float
    cems_key: str


def table_value(path: Path, first_cell: str, column: int) -> float:
    """The first number in ``column`` of the first Markdown table row whose first cell is ``first_cell``."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > column and cells[0] == first_cell:
            match = re.search(r"-?\d+\.\d+", cells[column])
            if match:
                return float(match.group())
    raise LookupError(f"{path.name}: no table row starting with {first_cell!r}")


def cems_tables() -> dict[tuple[str, bool], dict[str, Any]]:
    """The rebuilt-stack score tables keyed by (unit, restricted to GFM-covered units)."""
    data = json.loads((REPORTS / "v12_cems_rebuilt.json").read_text(encoding="utf-8"))
    return {(t["unit"], t["gfm_common"]): t for t in data["tables"]}


def points() -> list[Point]:
    s1f11, ks = REPORTS / "v11_sen1floods11_table.md", REPORTS / "v11_kurosiwo_table.md"
    pool, erratum = REPORTS / "v12_pool_cv.md", REPORTS / "v1_1_erratum.md"
    loeo, five_fold, pooled = "leave-one-event-out", "grouped five-fold", "grouped five-fold, 1 seed"
    return [
        Point("Fusion", "Sen1Floods11", loeo, table_value(s1f11, "9", 2), "s1f11_r9"),
        Point("TerraMind-small, corrected", "Sen1Floods11", loeo,
              table_value(erratum, "TerraMind, Sen1Floods11 leave-one-event-out", 2), "s1f11_tm_fixed"),
        Point("SAR U-Net", "Sen1Floods11", loeo, table_value(s1f11, "4", 2), "s1f11_r4"),
        Point("Fusion (v1.1 best)", "Kuro Siwo", five_fold, table_value(ks, "9", 2), "v11"),
        Point("SAR U-Net", "Kuro Siwo", five_fold, table_value(ks, "4", 2), "ks_r4"),
        Point("v1.2: fusion + seasonal context", "Kuro Siwo + GEOID-Flood", pooled,
              table_value(pool, "row21_fusion_context_seasonal", 2), "ours"),
        Point("TerraMind-base + LoRA (fair)", "Kuro Siwo + GEOID-Flood", pooled,
              table_value(pool, "cmp_terramind_base_fair", 2), "tm_fair"),
    ]


def _style() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                         "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def fig_cv_vs_cems(tables: dict[tuple[str, bool], dict[str, Any]], out: Path) -> None:
    models = tables[("activation", False)]["models"]
    rows = points()
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.55))
    for i, p in enumerate(rows):
        y = len(rows) - 1 - i
        cems = models[p.cems_key]["iou_tau"]
        colour = TRAINING_COLOURS[p.training]
        ax.annotate("", xy=(cems, y), xytext=(p.cv_iou, y),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0.9, "shrinkA": 3.2, "shrinkB": 3.2,
                                "mutation_scale": 6})
        ax.plot(p.cv_iou, y, "o", mfc="white", mec=colour, mew=1.0, ms=4.2, zorder=3)
        ax.plot(cems, y, "o", color=colour, ms=4.2, zorder=3)
        left, right = sorted((p.cv_iou, cems))
        ax.text(left - 0.012, y, f"{left:.3f}", ha="right", va="center", fontsize=6.3)
        ax.text(right + 0.012, y, f"{right:.3f}", ha="left", va="center", fontsize=6.3)
        print(f"cv_vs_cems: {p.label} [{p.training}] CV {p.cv_iou:.3f} -> CEMS {cems:.3f}")
    baseline = models["r2"]["iou_tau"]
    ax.axvline(baseline, color="0.45", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(baseline, len(rows) - 0.35, f"physics baseline\non CEMS ({baseline:.3f})", ha="center", va="bottom",
            fontsize=6, color="0.3", bbox={"fc": "white", "ec": "none", "pad": 0.6})
    print(f"cv_vs_cems: physics baseline on CEMS {baseline:.3f}")
    ax.set_yticks(range(len(rows)), [p.label for p in reversed(rows)])
    for tick, p in zip(ax.get_yticklabels(), reversed(rows), strict=True):
        tick.set_color(TRAINING_COLOURS[p.training])
    ax.set_xlim(0.18, 0.70)
    ax.set_ylim(-0.6, len(rows) + 0.55)
    ax.set_xlabel("Macro IoU at the model's own threshold $\\tau^*$")
    # Column-major fill: the training sets in the first column, the two scores in the second.
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"trained on {t}") for t, c in TRAINING_COLOURS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.2", ms=4, label="cross-validation"),
                plt.Line2D([], [], marker="o", ls="", color="0.2", ms=4, label="38 CEMS activations")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.36, -0.2), frameon=False, ncols=2,
              handlelength=1.4, columnspacing=1.6, borderaxespad=0.0)
    fig.savefig(out / "cv_vs_cems.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _per_unit(table: dict[str, Any], key: str) -> dict[str, float]:
    return {unit: v["iou_tau"] for unit, v in table["models"][key]["per_unit"].items() if v["iou_tau"] is not None}


def fig_per_activation(tables: dict[tuple[str, bool], dict[str, Any]], out: Path) -> None:
    panels = [(tables[("activation", False)], "tm_fair", "TerraMind-base + LoRA (fair)"),
              (tables[("activation", False)], "v11", "v1.1 best (fusion, Kuro Siwo)"),
              (tables[("activation", True)], "gfm", "Copernicus GFM (ensemble)")]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_IN, 2.35), sharey=True)
    for ax, (table, key, name), letter in zip(axes, panels, "abc", strict=True):
        ours, other = _per_unit(table, "ours"), _per_unit(table, key)
        units = sorted(set(ours) & set(other))
        xs, ys = [other[u] for u in units], [ours[u] for u in units]
        wins = sum(y > x for x, y in zip(xs, ys, strict=True))
        gap = table["models"]["ours"]["iou_tau"] - table["models"][key]["iou_tau"]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.scatter(xs, ys, s=11, color="#CC79A7", edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{name}: IoU at $\\tau^*$")
        ax.set_title(f"({letter}) K = {len(units)} activations", loc="left")
        ax.text(0.04, 0.95, f"v1.2 higher on {wins}/{len(units)}\nmacro gap {gap:+.3f}", va="top", fontsize=6.5)
        print(f"per_activation: vs {key}: K={len(units)} v1.2 higher on {wins}/{len(units)}, macro gap {gap:+.4f}")
    axes[0].set_ylabel("v1.2 (seasonal context): IoU at $\\tau^*$")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "per_activation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def quicklooks(out: Path) -> None:
    manifest = json.loads((ROOT / "assets" / "arena-manifest.json").read_text(encoding="utf-8"))
    pair = next(p for p in manifest["pairs"] if p["key"] == QUICKLOOK_PAIR)
    for layer in QUICKLOOK_LAYERS:
        source = ROOT / "assets" / f"arena-{QUICKLOOK_PAIR}-{layer}.webp"
        image = Image.open(source).convert("RGB")
        image.thumbnail((QUICKLOOK_SIDE, QUICKLOOK_SIDE), Image.Resampling.LANCZOS)
        image.save(out / f"emsr657_{layer}.png", optimize=True)
        iou = pair["layers"][layer]["iou"]
        print(f"quicklook: {layer}: {pair['layers'][layer]['name']}, tile IoU "
              f"{'—' if iou is None else f'{iou:.3f}'}")
    print(f"quicklook: tile {pair['tile']}, product {pair['product_time']} ({', '.join(pair['sensors'])}), "
          f"pass {pair['pass_time']}, flood {pair['flood_km2']} km²")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    tables = cems_tables()
    fig_cv_vs_cems(tables, OUT)
    fig_per_activation(tables, OUT)
    quicklooks(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
