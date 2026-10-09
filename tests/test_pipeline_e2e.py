"""The whole grid in miniature, on CPU: benchmark -> splits -> baseline -> train -> evaluate
-> row 11 -> ablation table -> product GeoTIFF. Every stage is the production code path; only
the sizes are small (two tiny events, 64 px chips, a 16-channel network, two epochs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("geopandas")

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = """
data:
  chip_root: "chips"
  chip_size: 64
  train_stride: 32
  eval_stride: 64
  num_workers: 0
  basin_buffer_m: 200.0
model:
  encoder_channels: [16, 32, 64, 128]
  decoder_channels: [64, 32, 16, 8]
  attention_heads: 2
  attention_head_dim: 16
  norm_groups: 4
train:
  epochs: 2
  warmup_epochs: 1
  micro_batch_size: 2
  grad_accum_steps: 2
  early_stopping_patience: 0
eval:
  n_bootstrap: 300
"""


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from floodfusion import splits
    from floodfusion.config import DataConfig, load_config
    from floodfusion.data.synthetic.benchmark import TINY_EVENTS, build_benchmark

    work = tmp_path_factory.mktemp("e2e")
    overlay = work / "overlay.yaml"
    overlay.write_text(OVERLAY)
    data_root = work / "data"
    build_benchmark(data_root, specs=TINY_EVENTS, data_cfg=DataConfig(chip_size=64, train_stride=32, eval_stride=64))
    assert splits.main(["--data-root", str(data_root), "--config", str(ROOT / "configs" / "row09_fusion_crossattn.yaml"),
                        "--overlay", str(overlay), "--allow-short-buffer", "--pixels-per-event", "3000"]) == 0
    cfg = load_config(ROOT / "configs" / "row09_fusion_crossattn.yaml", overlays=[overlay])
    return work, data_root, overlay, cfg


@pytest.fixture(scope="module")
def trained(pipeline):
    from floodfusion.train import fit, load_training_data

    work, data_root, overlay, cfg = pipeline
    run_dir = work / "runs" / cfg.name / "seed42"
    record = fit(load_training_data(cfg, data_root), run_dir, device=torch.device("cpu"),
                 provenance={"data_root": str(data_root), "overlays": [str(overlay)]})
    return run_dir, record


def test_training_writes_a_complete_run(trained) -> None:
    run_dir, record = trained
    assert record["status"] == "finished" and record["epochs_run"] == 2
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    epoch = json.loads(lines[-1])
    for key in ("loss_focal", "loss_dice", "gate_mean_with_optical", "lr", "val_best_macro_iou", "optimizer_steps"):
        assert key in epoch
    assert 0.0 < record["pi_train"] < 1.0 and 0.0 < record["pi_deploy"] < 1.0
    assert record["vram_prediction"]["predicted_bytes"] > 0 and record["manifest_sha256"]


def test_resume_after_a_crash(pipeline, monkeypatch) -> None:
    """Kill the run inside epoch 1; resuming from last.pt finishes it with one log line per epoch."""
    from floodfusion import train as train_module

    work, data_root, _, cfg = pipeline
    data = train_module.load_training_data(cfg, data_root)
    run_dir = work / "runs" / "crash" / "seed42"
    real_epoch = train_module.train_one_epoch
    calls = {"n": 0}

    def crashing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(train_module, "train_one_epoch", crashing)
    with pytest.raises(KeyboardInterrupt):
        train_module.fit(data, run_dir, device=torch.device("cpu"), provenance={"data_root": str(data_root)})
    monkeypatch.setattr(train_module, "train_one_epoch", real_epoch)
    record = train_module.fit(data, run_dir, device=torch.device("cpu"), resume=run_dir / "last.pt",
                              provenance={"data_root": str(data_root)})
    epochs = [json.loads(line)["epoch"] for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert record["epochs_run"] == 2 and epochs == [0, 1]
    assert record["train_seconds"] > 0


def test_resume_refuses_a_foreign_checkpoint(pipeline, trained) -> None:
    from dataclasses import replace

    from floodfusion.train import fit, load_training_data

    _, data_root, _, cfg = pipeline
    run_dir, _ = trained
    other = replace(cfg, train=replace(cfg.train, epochs=3))
    with pytest.raises(ValueError, match="different configuration"):
        fit(load_training_data(other, data_root), run_dir.parent / "foreign", device=torch.device("cpu"),
            resume=run_dir / "last.pt")


def test_evaluation_calibrates_on_validation_and_scores_test(trained) -> None:
    from floodfusion.evaluate import evaluate_run

    run_dir, _ = trained
    result = evaluate_run(run_dir, device="cpu")
    assert result["split"] == "test" and result["tau_star"] == pytest.approx(0.25)
    assert result["calibration"]["method"] in {"temperature", "platt"}
    assert set(result["per_basin"]) and all(0.0 <= v["n_flood"] <= v["n_valid"] for v in result["per_basin"].values())
    lo, hi = result["macro_iou_ci"]
    assert lo <= hi
    assert result["macro_iou_oracle"] + 1e-9 >= max(result["macro_iou"], result["macro_iou_val_selected"]) - 1e-9 or \
        np.isnan(result["macro_iou"])
    assert (run_dir / "eval_test.json").is_file() and (run_dir / "figures" / "pr_test.png").is_file()


def test_test_split_is_evaluated_once(trained) -> None:
    from floodfusion.evaluate import evaluate_run

    run_dir, _ = trained
    with pytest.raises(FileExistsError):
        evaluate_run(run_dir, device="cpu", figures=False)


def test_sar_only_never_overwrites_the_source_run(trained) -> None:
    from floodfusion.evaluate import evaluate_run

    run_dir, _ = trained
    with pytest.raises(ValueError, match="out-dir"):
        evaluate_run(run_dir, device="cpu", force_sar_only=True)


def test_row11_is_row9_weights_evaluated_sar_only(trained) -> None:
    from floodfusion.evaluate import evaluate_run

    run_dir, _ = trained
    out = run_dir.parent.parent / "row11_fusion_eval_sar_only" / "seed42"
    result = evaluate_run(run_dir, device="cpu", force_sar_only=True, out_dir=out)
    derived = json.loads((out / "run.json").read_text())
    assert result["ablation_row"] == 11 and result["force_sar_only"]
    assert derived["source_run"] == str(run_dir) and derived["trained"] is False


def test_baseline_row_and_table(pipeline, trained) -> None:
    from floodfusion.baseline_runner import run_baseline_row
    from floodfusion.config import load_config
    from floodfusion.reporting import build_ablation_table

    work, data_root, overlay, _ = pipeline
    cfg2 = load_config(ROOT / "configs" / "row02_baseline_adaptive.yaml", overlays=[overlay])
    result = run_baseline_row(cfg2, data_root, work / "runs" / cfg2.name / "baseline", attribution=True)
    assert set(result["mask_attribution"]) >= {"all_masks", "no_slope", "no_hand", "or_rule"}
    text = build_ablation_table(work / "runs", out=work / "reports" / "table.md", allow_dirty=True, n_boot=300)
    assert "| 2 | Baseline: adaptive threshold" in text and "| 9 | **Fusion, cross-attention" in text
    assert "SYNTHETIC BENCHMARK" in text


def test_product_geotiff(pipeline, trained) -> None:
    import rasterio

    from floodfusion import predict

    _, data_root, _, _ = pipeline
    run_dir, _ = trained
    stack = sorted((data_root / "events").glob("*/stack.tif"))[0]
    out = run_dir.parent / "product.tif"
    assert predict.main(["--run-dir", str(run_dir), "--stack", str(stack), "--out", str(out), "--device", "cpu"]) == 0
    with rasterio.open(out) as src, rasterio.open(stack) as ref:
        assert src.count == 2 and src.dtypes == ("uint16", "uint16") and src.shape == ref.shape
        classes = src.read(1)
    assert set(np.unique(classes)) <= {0, 1, 2, 255}
    with rasterio.open(out.with_name("product_prob.tif")) as src:
        prob = src.read(1)
    finite = prob[np.isfinite(prob)]
    assert finite.size and finite.min() >= 0.0 and finite.max() <= 1.0
