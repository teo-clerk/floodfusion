"""What the fusion gate learned: its mean over clear vs cloud-masked pixels, on the stress scenes.

The gate g in ``F = F_s + g * Attn(Q_optical, K_SAR, V_SAR)`` starts at sigmoid(bias_init). This
reports, per stress cloud scene (built by ``make stress``), the mean gate over clear pixels
(optical quality > 0.5) and over cloud-masked ones, and the share of each above 0.5.
Writes ``reports/gate_diagnostic.md``. CPU by default, so it can run beside a training job.

Run: ``uv run python scripts/gate_diagnostic.py --runs-root runs --work data/stress``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SCENES = ("STRESS_cloud_fraction=0.0_s9101", "STRESS_cloud_fraction=0.4_s9101", "STRESS_cloud_fraction=0.8_s9101")


def gate_by_quality(model: dict, records: list, device) -> dict[str, float]:
    """Pool the per-pixel gate of every chip over clear and cloud-masked pixels."""
    import torch

    from floodfusion.data.dataset import FloodChipDataset
    from floodfusion.train import model_inputs

    cfg = model["cfg"]
    ds = FloodChipDataset(records, cfg.norm, cfg.data, cfg.grid)
    clear, cloud = [], []
    for i in range(len(ds)):
        s = ds[i]
        batch = {k: (v.unsqueeze(0).to(device) if hasattr(v, "unsqueeze") else [v]) for k, v in s.items()}
        with torch.inference_mode():
            _, gate = model["model"](**model_inputs(batch, batch["optical_available"]))
        q = s["optical_quality"].numpy()[0]
        g = torch.nn.functional.interpolate(gate.float(), size=q.shape, mode="nearest")[0, 0].cpu().numpy()
        clear.append(g[q > 0.5])
        cloud.append(g[q <= 0.5])
    c, u = np.concatenate(clear), np.concatenate(cloud)
    nan = float("nan")
    return {"chips": len(ds), "clear_mean": float(c.mean()) if c.size else nan,
            "cloud_mean": float(u.mean()) if u.size else nan,
            "clear_open": float((c > 0.5).mean()) if c.size else nan,
            "cloud_open": float((u > 0.5).mean()) if u.size else nan}


def main() -> int:
    """Load row 9 (seed 42), score the gate on each cloud scene, write the report."""
    import torch

    from floodfusion.ablation import ROW_CONFIGS
    from floodfusion.data.records import load_manifest
    from floodfusion.stress import _learned

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--work", type=Path, default=Path("data/stress"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("reports/gate_diagnostic.md"))
    args = parser.parse_args()
    model = _learned(args.runs_root / ROW_CONFIGS[9] / "seed42", torch.device(args.device), False)
    cfg = model["cfg"]
    init = 1.0 / (1.0 + math.exp(-cfg.model.gate_bias_init))
    rows = []
    for scene in SCENES:
        manifest = args.work / scene / Path(cfg.data.chip_root).name / cfg.data.manifest_name
        records = [r for r in load_manifest(manifest) if r.eval_grid]
        rows.append((scene.split("=", 1)[1].rsplit("_s", 1)[0], gate_by_quality(model, records, args.device)))
    fmt = lambda v: "—" if v != v else f"{v:.3f}"  # noqa: E731
    lines = [
        "# What the fusion gate learned (row 9, seed 42, synthetic stress scenes)", "",
        f"The gate starts at sigmoid(bias_init) = {init:.3f}. Clear = optical quality > 0.5.", "",
        "| Cloud fraction | Chips | Mean gate, clear | Mean gate, cloud-masked | Clear pixels g > 0.5 "
        "| Cloud-masked pixels g > 0.5 |",
        "| --- | --- | --- | --- | --- | --- |",
        *[f"| {name} | {r['chips']} | {fmt(r['clear_mean'])} | {fmt(r['cloud_mean'])} | {fmt(r['clear_open'])} "
          f"| {fmt(r['cloud_open'])} |" for name, r in rows], "",
        "Regenerate: `uv run python scripts/gate_diagnostic.py` (after `make demo` and `make stress`).", "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
