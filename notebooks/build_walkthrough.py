"""Write ``notebooks/walkthrough.ipynb`` from the cells below (run: ``python notebooks/build_walkthrough.py``).

The notebook is generated, not hand-edited, so its source stays reviewable in a diff. Execute
it with ``jupyter nbconvert --execute`` or ``tests/test_notebook.py`` (which does so in CI).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
    ("md", """# floodfusion — a walkthrough on one small synthetic event

Everything below runs **offline on a CPU in about a minute**: a physics-based synthetic flood event,
its analysis-ready stack, a basin-blocked split, a (deliberately tiny) cross-attention fusion model,
calibration on validation, and the operational product with its exclusion mask. Every call is the
production code path; only the sizes are small.

The real-data results (Sen1Floods11 re-ingested, the EMSR664 Forlì case study) are in `reports/`."""),
    ("code", """import json, tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from floodfusion import splits
from floodfusion.config import DataConfig, load_config
from floodfusion.data.chipping import read_event_stack
from floodfusion.data.encoding import ChipBand
from floodfusion.data.synthetic.benchmark import TINY_EVENTS, build_benchmark

torch.manual_seed(0)
plt.rcParams.update({"figure.dpi": 60, "savefig.dpi": 60})   # keep the committed notebook small
ROOT = Path.cwd() if (Path.cwd() / "configs").is_dir() else Path.cwd().parent
WORK = Path(tempfile.mkdtemp(prefix="floodfusion_walkthrough_"))
data_root = WORK / "data"
result = build_benchmark(data_root, specs=TINY_EVENTS, data_cfg=DataConfig(chip_size=64, train_stride=32, eval_stride=64))
print(len(result.records), "chips from", [e["event_id"] for e in result.events])"""),
    ("md", """## 1. The analysis-ready stack

One grid, sixteen bands: post- and pre-event γ⁰ (VV, VH), their change, Sentinel-2 B3/B8/B11 and MNDWI,
slope, HAND, local incidence, the label, the validity bit field, the optical quality and land cover.
Flood water is dark in the post image and *darker than before*; terrain decides where radar sees nothing."""),
    ("code", """stack, grid, _ = read_event_stack(data_root / "events" / "T1_lowland" / "stack.tif")
db = lambda b: np.where(stack[b] == -32768, np.nan, stack[b] / 100.0)
panels = [(db(ChipBand.VV_PRE), "VV pre (dB)", "gray"), (db(ChipBand.VV_POST), "VV post (dB)", "gray"),
          (db(ChipBand.DELTA_VV), "ΔVV post − pre (dB)", "RdBu"), (stack[ChipBand.MNDWI] / 1e4, "MNDWI", "BrBG"),
          (np.where(stack[ChipBand.LABEL] < 0, np.nan, stack[ChipBand.LABEL]), "label (flood)", "Blues"),
          (stack[ChipBand.VALIDITY], "validity bit field", "magma")]
fig, axes = plt.subplots(1, 6, figsize=(18, 3.6))
for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, cmap=cmap); ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout()"""),
    ("md", """## 2. A split that does not flatter

Chips are grouped by hydrological basin; no basin is in two splits, and chips near a boundary are buffered
out. A random chip split would put overlapping windows of the same river on both sides — the leakage row
of the ablation measures exactly how much that inflates a score."""),
    ("code", """overlay = WORK / "overlay.yaml"
overlay.write_text('''
data: {chip_root: chips, chip_size: 64, train_stride: 32, eval_stride: 64, num_workers: 0, basin_buffer_m: 200.0}
model: {encoder_channels: [16, 32, 64, 128], decoder_channels: [64, 32, 16, 8], attention_heads: 2,
        attention_head_dim: 16, norm_groups: 4}
train: {epochs: 3, warmup_epochs: 1, micro_batch_size: 2, grad_accum_steps: 2, early_stopping_patience: 0}
eval: {n_bootstrap: 300}
''')
config = ROOT / "configs" / "row09_fusion_crossattn.yaml"
splits.main(["--data-root", str(data_root), "--config", str(config), "--overlay", str(overlay),
             "--allow-short-buffer", "--pixels-per-event", "3000"])
frozen = json.loads((data_root / "chips" / "splits.json").read_text())
print({k: len(frozen[f"{k}_chips"]) for k in ("train", "val", "test")}, "| test basins:", frozen["test_basins"])"""),
    ("md", """## 3. Train, calibrate, score

The model: two encoders, cross-attention with queries from optical and keys/values from SAR, a per-pixel
quality gate, and a gated residual on the SAR stream (with the gate closed, the output is the SAR-only
path, bit for bit). Calibration — prior correction and temperature (Platt if clearly better) — is fitted
on validation, the threshold τ* = 1/(1+r) follows from a stated cost ratio, and a conformal threshold
bounds the expected per-chip miss rate."""),
    ("code", """from floodfusion.evaluate import evaluate_run
from floodfusion.train import fit, load_training_data

cfg = load_config(config, overlays=[overlay])
run_dir = WORK / "runs" / cfg.name / "seed42"
fit(load_training_data(cfg, data_root), run_dir, device=torch.device("cpu"), provenance={"data_root": str(data_root)})
ev = evaluate_run(run_dir, data_root=data_root, device=torch.device("cpu"), figures=False)
print(f"test macro IoU at tau*={ev['tau_star']:.2f}: {ev['macro_iou']:.3f} (raw scores: {ev['macro_iou_raw_tau_star']:.3f})")
print("calibration:", {k: round(v, 3) if isinstance(v, float) else v for k, v in ev["calibration"].items()
                       if k in ("method", "temperature", "offset")})
print("conformal:", {k: ev["conformal"][k] for k in ("alpha", "tau", "n", "test_mean_chip_fnr")})"""),
    ("md", """## 4. The product

Seven Cloud-Optimized GeoTIFFs with the structure of the Copernicus Global Flood Monitoring product, and a
STAC Item that links every input scene. Excluded pixels (shadow, layover, dense forest or urban) keep
their likelihood: the mask says *do not trust this*, it does not erase the evidence."""),
    ("code", """from floodfusion import predict

stack_path = data_root / "events" / "T1_lowland" / "stack.tif"
predict.main(["--run-dir", str(run_dir), "--stack", str(stack_path), "--out", str(WORK / "product.tif"),
              "--package", str(WORK / "package"), "--device", "cpu"])
import rasterio
layers = {n: rasterio.open(WORK / "package" / f"{n}.tif").read(1) for n in ("likelihood", "observed_flood", "exclusion_mask")}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (name, img) in zip(axes, layers.items()):
    ax.imshow(np.where(img == 255, np.nan, img), cmap="viridis"); ax.set_title(name); ax.axis("off")
plt.tight_layout()
item = json.loads((WORK / "package" / "item.json").read_text())
print(item["stac_extensions"])
print(sorted(item["assets"]))"""),
]


def build(path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    print(build(Path(__file__).with_name("walkthrough.ipynb")))
