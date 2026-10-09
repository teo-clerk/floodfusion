"""``load_config``: the extends chain, overlays, overrides, strict schema and hashing.

Complements ``test_configs.py`` (which validates the YAML grid with its own resolver) by
exercising the package's loader — the one every entry point actually calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from floodfusion.config import (
    ExperimentConfig,
    config_from_dict,
    config_sha256,
    config_to_dict,
    load_config,
    resolve_config_dict,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path: Path) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.train.amp_dtype == "bfloat16"


def test_base_pins_the_raster_chip_source() -> None:
    """Reported runs must never fall back to simulated pixels."""
    assert load_config(CONFIG_DIR / "row09_fusion_crossattn.yaml").data.chip_source == "raster"


def test_tuples_and_paths_are_converted() -> None:
    cfg = load_config(CONFIG_DIR / "row09_fusion_crossattn.yaml")
    assert isinstance(cfg.model.encoder_channels, tuple)
    assert isinstance(cfg.data.chip_root, Path)
    assert isinstance(cfg.output_dir, Path)


def test_row13_reads_its_own_split_file() -> None:
    cfg = load_config(CONFIG_DIR / "row13_fusion_random_split.yaml")
    assert cfg.data.split_strategy == "random_chip"
    assert cfg.data.split_file_name == "splits_random_chip.json"


def test_overrides_are_parsed_as_yaml() -> None:
    cfg = load_config(
        CONFIG_DIR / "row09_fusion_crossattn.yaml",
        overrides=["train.epochs=3", "train.warmup_epochs=1", "model.modality_dropout_p=0.8"],
    )
    assert cfg.train.epochs == 3 and isinstance(cfg.train.epochs, int)
    assert cfg.model.modality_dropout_p == pytest.approx(0.8)


def test_overlay_is_deep_merged(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("train:\n  epochs: 7\n  warmup_epochs: 2\ndata:\n  num_workers: 0\n")
    cfg = load_config(CONFIG_DIR / "row04_unet_sar_only.yaml", overlays=[overlay])
    assert cfg.train.epochs == 7
    assert cfg.data.num_workers == 0
    assert cfg.train.micro_batch_size == 8, "fields not in the overlay keep the row's value"


def test_overlay_cannot_extend(tmp_path: Path) -> None:
    overlay = tmp_path / "bad.yaml"
    overlay.write_text("extends: base.yaml\n")
    with pytest.raises(ValueError, match="extends"):
        load_config(CONFIG_DIR / "row09_fusion_crossattn.yaml", overlays=[overlay])


def test_all_unknown_keys_are_reported_at_once() -> None:
    """A typo must fail loudly — and every typo, not just the first."""
    raw = resolve_config_dict(CONFIG_DIR / "base.yaml")
    raw["train"]["epoch"] = 3
    raw["model"]["fusion_stag"] = 2
    raw["typo_section"] = {}
    with pytest.raises(ValueError) as excinfo:
        config_from_dict(raw)
    message = str(excinfo.value)
    assert "epoch" in message and "fusion_stag" in message and "typo_section" in message


def test_post_init_guards_still_apply() -> None:
    with pytest.raises(ValueError, match=r"fp16|float16|amp_dtype"):
        load_config(CONFIG_DIR / "base.yaml", overrides=["train.amp_dtype=float16"])
    with pytest.raises(ValueError, match="random_chip"):
        load_config(CONFIG_DIR / "row09_fusion_crossattn.yaml",
                    overrides=["data.split_strategy=random_chip"])


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(CONFIG_DIR / "does_not_exist.yaml")


def test_hash_is_stable_and_sensitive() -> None:
    a = resolve_config_dict(CONFIG_DIR / "row09_fusion_crossattn.yaml")
    b = resolve_config_dict(CONFIG_DIR / "row09_fusion_crossattn.yaml")
    c = resolve_config_dict(CONFIG_DIR / "row10_fusion_no_modality_dropout.yaml")
    assert config_sha256(a) == config_sha256(b)
    assert config_sha256(a) != config_sha256(c)
    assert len(config_sha256(a)) == 64


def test_round_trip_through_plain_dict() -> None:
    cfg = load_config(CONFIG_DIR / "row09_fusion_crossattn.yaml")
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_chip_size_must_suit_the_encoder() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        load_config(CONFIG_DIR / "base.yaml", overrides=["data.chip_size=250"])
