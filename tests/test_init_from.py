"""v1.3's fine-tuning start (``train --init-from``; protocol v1.3 §3): a run begins from another run's weights."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import config_sha256, config_to_dict, load_config  # noqa: E402
from floodfusion.models.zoo import build_model  # noqa: E402
from floodfusion.runs import write_run_record  # noqa: E402
from floodfusion.train import TrainingData, adopt_init, fit, init_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL = """
model:
  encoder_channels: [16, 32, 64, 128]
  decoder_channels: [64, 32, 16, 8]
  attention_heads: 2
  attention_head_dim: 16
  norm_groups: 4
"""


@pytest.fixture
def cfg(tmp_path: Path):
    overlay = tmp_path / "small.yaml"
    overlay.write_text(SMALL)
    return load_config(ROOT / "configs" / "row09_fusion_crossattn.yaml", overlays=[overlay])


def _source_run(tmp_path: Path, cfg, *, status: str = "finished", foreign: bool = False) -> tuple[Path, dict]:
    torch.manual_seed(0)
    model = build_model(cfg.model)
    run = tmp_path / "source" / "seed42"
    sha = config_sha256(config_to_dict(cfg))
    write_run_record(run, {"status": status, "config": config_to_dict(cfg), "config_sha256": sha,
                           "norm_stats_file": "data/pool/chips/cv/all__all.norm_stats.json",
                           "architecture": cfg.model.architecture})
    torch.save({"model": model.state_dict(), "epoch": 3, "val_macro_iou": 0.5,
                "config_sha256": "other" if foreign else sha}, run / "best.pt")
    return run, model.state_dict()


def _data(cfg) -> TrainingData:
    return TrainingData(train=(), val=(), test=(), cfg=cfg, manifest_sha256="m", manifest_provenance={},
                        split_path=Path("cv/kfold__fold0.json"), split_strategy="leave_one_basin_out",
                        norm_stats_path=Path("cv/kfold__fold0.norm_stats.json"))


def test_init_weights_loads_the_source_checkpoint_exactly(tmp_path: Path, cfg) -> None:
    run, state = _source_run(tmp_path, cfg)
    torch.manual_seed(1)
    model = build_model(cfg.model)
    info = init_weights(model, run, device=torch.device("cpu"))
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert info == {"run": str(run), "epoch": 3,
                    "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()}


def test_init_weights_refuses_an_unfinished_run_or_a_foreign_checkpoint(tmp_path: Path, cfg) -> None:
    unfinished, _ = _source_run(tmp_path / "a", cfg, status="running")
    with pytest.raises(ValueError, match="not 'finished'"):
        init_weights(build_model(cfg.model), unfinished, device=torch.device("cpu"))
    foreign, _ = _source_run(tmp_path / "b", cfg, foreign=True)
    with pytest.raises(ValueError, match="different configuration"):
        init_weights(build_model(cfg.model), foreign, device=torch.device("cpu"))


def test_adopt_init_takes_the_source_runs_normalisation(tmp_path: Path, cfg) -> None:
    run, _ = _source_run(tmp_path, cfg)
    fold_norm = replace(cfg.norm, sar_mean_db=(0.0,) * len(cfg.norm.sar_mean_db))
    adopted = adopt_init(_data(replace(cfg, norm=fold_norm)), run)
    assert adopted.cfg.norm == cfg.norm
    assert adopted.norm_stats_path == Path("data/pool/chips/cv/all__all.norm_stats.json")
    assert adopted.cfg.train == cfg.train                     # only the normalisation changes


def test_adopt_init_refuses_a_different_network(tmp_path: Path, cfg) -> None:
    run, _ = _source_run(tmp_path, cfg)
    wider = replace(cfg, model=replace(cfg.model, attention_heads=4))
    with pytest.raises(ValueError, match="network"):
        adopt_init(_data(wider), run)


def test_fit_refuses_to_start_from_a_run_without_its_normalisation(tmp_path: Path, cfg) -> None:
    run, _ = _source_run(tmp_path, cfg)
    fold_norm = replace(cfg.norm, sar_mean_db=(0.0,) * len(cfg.norm.sar_mean_db))
    with pytest.raises(ValueError, match="adopt_init"):
        fit(_data(replace(cfg, norm=fold_norm)), tmp_path / "out", device=torch.device("cpu"), init_from=run)
    assert not (tmp_path / "out").exists()
