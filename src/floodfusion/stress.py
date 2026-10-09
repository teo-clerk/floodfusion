r"""Where it breaks: one physical factor at a time, on landscapes no model has seen.

Real data mixes every confounder at once; the synthetic generator can move one and hold
the rest. Each sweep renders fresh scenes — a **new landscape seed**, never one of the
benchmark's training events — from the same frozen generator, varying only:

``wind_speed_ms``     Bragg roughening of open water (docs/01 section 6): water stops being dark
``cloud_fraction``    optical availability (the fusion model's whole premise)
``incidence``         near/far incidence of the swath (contrast and shadow change with it)
``harvest_fraction``  cropland darkened between dates — the false alarm the physics baseline
                      was caught on (the row 1 vs row 2 diagnosis)

and scores the physics baseline (row 2), the SAR-only U-Net (row 4), the fusion model
(row 9) and the fusion model run SAR-only (row 11) at their calibrated operating points.
It also draws the **gate** of the fusion model on a partly clouded scene: the one picture
that shows R2 and R3 at work — the gate opens over clear optical pixels and closes under
cloud.

Run as (``make stress``)::

    python -m floodfusion.stress --runs-root runs --out reports/stress.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FACTORS", "base_spec", "main", "variant"]

LOGGER = logging.getLogger(__name__)

#: A lowland scene like the benchmark's E1 — but a landscape seed no training event used.
_STRESS_SEED = 9101
FACTORS: dict[str, tuple[Any, ...]] = {
    "wind_speed_ms": (0.5, 2.0, 4.0, 6.0, 8.0, 10.0),
    "cloud_fraction": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "incidence": ((30.0, 34.0), (34.0, 38.0), (38.0, 42.0), (42.0, 46.0)),
    "harvest_fraction": (0.0, 0.1, 0.2, 0.3),
}


def base_spec() -> Any:
    from floodfusion.data.synthetic.landscape import EventSpec

    return EventSpec("STRESS", seed=_STRESS_SEED, n_basins=2, basin_width_px=768, height_px=1024,
                     terrain="lowland", forest_cover=0.15, town=True, runway=True, wind_speed_ms=1.0,
                     orbit_state="descending", relative_orbit=117, optical="cloudy", cloud_fraction=0.0,
                     flood_depth_m=(1.2, 1.6), harvest_fraction=0.08)



def variant(factor: str, value: Any) -> Any:
    """The base scene with ONE factor changed; everything else, the seed included, is fixed."""
    base = base_spec()
    tag = f"{factor}={value if not isinstance(value, tuple) else '-'.join(f'{v:g}' for v in value)}"
    if factor == "incidence":
        return replace(base, event_id=f"STRESS_{tag}", incidence_near_deg=value[0], incidence_far_deg=value[1])
    if factor not in FACTORS:
        raise ValueError(f"unknown factor {factor!r}; known: {sorted(FACTORS)}")
    return replace(base, event_id=f"STRESS_{tag}", **{factor: value})


# ---------------------------------------------------------------------------
# Models at their operating points
# ---------------------------------------------------------------------------
def _learned(run_dir: Path, device: Any, sar_only: bool) -> dict[str, Any]:
    from floodfusion.evaluate import _load

    evaluation = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    cfg, _, model, _, _ = _load(Path(evaluation.get("run_dir", run_dir)) if sar_only else run_dir, None, device, sar_only)
    cal = evaluation["calibration"]
    return {"cfg": cfg, "model": model, "temperature": cal["temperature"], "offset": cal["offset"], "sar_only": sar_only}


def _score_learned(m: Mapping[str, Any], records: Sequence[Any], device: Any) -> Any:
    from floodfusion.evaluate import _loader
    from floodfusion.metrics import sum_confusions
    from floodfusion.train import EVAL_CONTEXT_PX, validate

    cfg = m["cfg"]
    tau = cfg.eval.decision_threshold
    result = validate(m["model"], _loader(records, cfg), device=device, cfg=cfg, thresholds=(tau,),
                      force_sar_only=m["sar_only"], crop_px=EVAL_CONTEXT_PX, temperature=m["temperature"],
                      offset=m["offset"])
    return sum_confusions([v[0] for v in result["per_basin"].values()])


def _score_baseline(run_dir: Path, configs_dir: Path, overlays: Sequence[Path], records: Sequence[Any]) -> Any:
    from floodfusion.baseline_runner import (
        _windows_by_event,
        flood_map,
        load_scene,
        score_windows,
        threshold_surface,
    )
    from floodfusion.config import load_config
    from floodfusion.metrics import sum_confusions

    evaluation = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    cfg = load_config(configs_dir / "row02_baseline_adaptive.yaml", overlays=overlays)
    cms = []
    for path, windows in _windows_by_event(records).items():
        scene = load_scene(path, cfg)
        surface, _ = threshold_surface(scene, cfg)
        pred = flood_map(scene, surface, cfg, offset_db=float(evaluation["operating_offset_db"]))
        cms += list(score_windows(pred, scene, windows).values())
    return sum_confusions(cms)


# ---------------------------------------------------------------------------
# The gate figure
# ---------------------------------------------------------------------------
def gate_figure(m: Mapping[str, Any], records: Sequence[Any], device: Any, out: Path) -> Path:
    """Optical (with cloud), the learned gate, VV post, prediction and label on one chip."""
    import torch

    from floodfusion.data.dataset import FloodChipDataset
    from floodfusion.figures import _pyplot
    from floodfusion.train import model_inputs

    cfg = m["cfg"]
    ds = FloodChipDataset(records, cfg.norm, cfg.data, cfg.grid)
    # The chip with the most even split between clear and clouded optical pixels.
    def mixed(i: int) -> float:
        q = ds[i]["optical_quality"].numpy()
        return -abs(float((q > 0.5).mean()) - 0.5)
    sample = ds[max(range(len(ds)), key=mixed)]
    batch = {k: (v.unsqueeze(0).to(device) if hasattr(v, "unsqueeze") else [v]) for k, v in sample.items()}
    with torch.inference_mode():
        logits, gate = m["model"](**model_inputs(batch, batch["optical_available"]))
    prob = torch.sigmoid((logits.float() + m["offset"]) / m["temperature"])[0, 0].cpu().numpy()
    g = torch.nn.functional.interpolate(gate.float(), size=prob.shape, mode="nearest")[0, 0].cpu().numpy()
    optical = sample["optical"].numpy()
    quality = sample["optical_quality"].numpy()[0]
    rgb = np.clip(np.stack([optical[2], optical[1], optical[0]], axis=-1), 0, 1)   # SWIR, NIR, green
    plt = _pyplot()
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    panels = (  # (image, title, colour map, (vmin, vmax))
        (rgb, "Sentinel-2 SWIR/NIR/green", None, (None, None)),
        (quality, "optical quality (SCL, dilated)", "gray", (0.0, 1.0)),
        (g, "learned gate g", "magma", (0.0, 1.0)),
        (sample["sar"].numpy()[0], "VV post (normalised)", "gray", (None, None)),
        (prob > cfg.eval.decision_threshold, "prediction at τ*  (label contour)", "Blues", (0, 1)),
    )
    for ax, (img, title, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if cmap == "magma":
            fig.colorbar(im, ax=ax, fraction=0.046)
    axes[-1].contour(sample["label"].numpy()[0], levels=[0.5], colors="crimson", linewidths=0.8)
    fig.suptitle("The gate opens where optical is clear and closes under cloud (R2/R3 by construction)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _plot_curves(results: Mapping[str, Mapping[str, Mapping[str, float]]], out: Path) -> Path:
    from floodfusion.figures import _pyplot

    plt = _pyplot()
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6), sharey=True)
    for ax, (factor, by_value) in zip(np.atleast_1d(axes), results.items()):
        labels = list(by_value)
        for model in next(iter(by_value.values())):
            ax.plot(range(len(labels)), [by_value[v][model] for v in labels], marker="o", label=model)
        ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8)
        ax.set_title(factor.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("IoU at τ*")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    from floodfusion.ablation import ROW_CONFIGS
    from floodfusion.data.records import load_manifest
    from floodfusion.data.synthetic.benchmark import build_benchmark
    from floodfusion.runs import write_json
    from floodfusion.train import resolve_device

    device = resolve_device(args.device)
    seed = f"seed{args.seed}"
    learned = {
        "row 4 U-Net SAR": _learned(args.runs_root / ROW_CONFIGS[4] / seed, device, False),
        "row 9 fusion": _learned(args.runs_root / ROW_CONFIGS[9] / seed, device, False),
        "row 11 fusion, SAR-only": _learned(args.runs_root / ROW_CONFIGS[11] / seed, device, True),
    }
    cfg = learned["row 9 fusion"]["cfg"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    gate_records = None
    for factor, values in FACTORS.items():
        results[factor] = {}
        for value in values:
            spec = variant(factor, value)
            root = args.work / spec.event_id
            manifest = root / Path(cfg.data.chip_root).name / cfg.data.manifest_name
            if not manifest.is_file():
                build_benchmark(root, specs=[spec], data_cfg=cfg.data, grid_cfg=cfg.grid, norm=cfg.norm)
            records = [r for r in load_manifest(manifest) if r.eval_grid]
            label = spec.event_id.split("=", 1)[1]
            scores = {"row 2 baseline": _score_baseline(args.runs_root / ROW_CONFIGS[2] / "baseline", args.configs_dir,
                                                        args.overlay, records).iou}
            scores.update({name: _score_learned(m, records, device).iou for name, m in learned.items()})
            results[factor][label] = scores
            LOGGER.info("%s: %s", spec.event_id, {k: round(v, 3) for k, v in scores.items()})
            if factor == "cloud_fraction" and value == 0.4:
                gate_records = records
    _plot_curves(results, args.assets / "stress_curves.png")
    if gate_records:
        gate_figure(learned["row 9 fusion"], gate_records, device, args.assets / "gate_map.png")
    lines = ["# Stress test — one physical factor at a time (synthetic, unseen landscape)", "",
             f"Fresh landscape seed {_STRESS_SEED} (no training event used it); every other parameter fixed. IoU at "
             "each model's calibrated operating point τ*; the baseline at its validation-chosen offset.", ""]
    for factor, by_value in results.items():
        models = list(next(iter(by_value.values())))
        lines += [f"## {factor.replace('_', ' ')}", "", "| value | " + " | ".join(models) + " |",
                  "| --- |" + " --- |" * len(models)]
        lines += [f"| {v} | " + " | ".join(f"{s[m]:.3f}" for m in models) + " |" for v, s in by_value.items()]
        lines.append("")
    lines += ["![stress curves](../assets/stress_curves.png)", "", "![gate](../assets/gate_map.png)", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.out.with_suffix(".json"), {"seed": _STRESS_SEED, "results": results})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-factor stress curves and the gate figure (synthetic).")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work", type=Path, default=Path("data/stress"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--out", type=Path, default=Path("reports/stress.md"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
