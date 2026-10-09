"""The frozen ablation grid: every YAML must resolve against the dataclass schema.

These tests pass on a fresh clone — the configs are data, not skeletons — and they exist
to stop the grid drifting away from ``MILESTONES.md`` §6 or from
:mod:`floodfusion.config`.

The three properties that matter most:

1. **Unknown keys are an error.** A silently ignored typo in a config key is a
   reproducibility failure: the run proceeds with a default you did not intend and the
   log records a value you never used.
2. **Every row of the grid is present exactly once**, so ``evaluate.py
   --ablation-table`` can regenerate all 13 rows from disk.
3. **Row 13 is the only configuration permitted to leak.** Any other row using
   ``random_chip`` would report an inflated number.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to validate the config grid")

from floodfusion import config as C

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

#: Top-level scalar keys permitted in a config file.
SCALAR_KEYS = {"name", "ablation_row", "output_dir"}

#: Mapping from YAML section name to the dataclass that models it.
SECTIONS: dict[str, type] = {
    "sensor": C.SensorConstants,
    "grid": C.GridConfig,
    "norm": C.NormalizationStats,
    "data": C.DataConfig,
    "model": C.ModelConfig,
    "baseline": C.BaselineConfig,
    "loss": C.LossConfig,
    "train": C.TrainConfig,
    "eval": C.EvalConfig,
}


# ---------------------------------------------------------------------------
# Resolution helpers (mirror what load_config must implement)
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path: Path, _depth: int = 0) -> dict[str, Any]:
    """Resolve an ``extends:`` chain into a flat mapping."""
    if _depth > 8:
        raise RecursionError(f"extends chain too deep at {path.name}; cycle?")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return raw
    return _deep_merge(_resolve(CONFIG_DIR / parent_name, _depth + 1), raw)


def _instantiate(raw: dict[str, Any]) -> C.ExperimentConfig:
    """Build an :class:`ExperimentConfig`, exercising every ``__post_init__`` guard."""
    kwargs: dict[str, Any] = {}
    for section, cls in SECTIONS.items():
        values = dict(raw.get(section, {}))
        # YAML lists become tuples; dataclasses declare tuple fields.
        values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()}
        if cls is C.DataConfig and "chip_root" in values:
            values["chip_root"] = Path(values["chip_root"])
        kwargs[section] = cls(**values)
    return C.ExperimentConfig(
        name=raw["name"],
        ablation_row=raw["ablation_row"],
        output_dir=Path(raw["output_dir"]),
        **kwargs,
    )


ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
ROW_CONFIGS = sorted(CONFIG_DIR.glob("row*.yaml"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_config_directory_is_populated() -> None:
    assert ALL_CONFIGS, f"no YAML configs found under {CONFIG_DIR}"


def test_base_config_exists() -> None:
    assert (CONFIG_DIR / "base.yaml").is_file()


def test_base_declares_every_section() -> None:
    """``base.yaml`` holds the defaults, so it must be complete."""
    raw = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    assert set(raw) == SCALAR_KEYS | set(SECTIONS)


def test_base_does_not_extend_anything() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    assert "extends" not in raw, "base.yaml is the root of the extends chain"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=_ids(ALL_CONFIGS))
class TestEveryConfig:
    def test_parses_as_yaml(self, path: Path) -> None:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

    def test_resolves_and_instantiates(self, path: Path) -> None:
        """The real check: it must survive every ``__post_init__`` guard."""
        _instantiate(_resolve(path))

    def test_has_no_unknown_top_level_keys(self, path: Path) -> None:
        raw = _resolve(path)
        unknown = set(raw) - SCALAR_KEYS - set(SECTIONS)
        assert not unknown, f"{path.name}: unknown top-level keys {sorted(unknown)}"

    def test_has_no_unknown_section_fields(self, path: Path) -> None:
        """A typo'd field name must fail loudly, not default silently."""
        raw = _resolve(path)
        for section, cls in SECTIONS.items():
            known = {f.name for f in dataclasses.fields(cls)}
            extra = set(raw.get(section, {})) - known
            assert not extra, f"{path.name}: [{section}] unknown fields {sorted(extra)}"

    def test_mandates_bfloat16(self, path: Path) -> None:
        """fp16 overflows the soft-Dice reduction (``docs/08`` section 5.1)."""
        assert _instantiate(_resolve(path)).train.amp_dtype == "bfloat16"

    def test_name_matches_filename(self, path: Path) -> None:
        """Keeps run directories traceable back to their config."""
        if path.name in {"base.yaml", "baseline_config.yaml"}:
            pytest.skip("base and the deprecation shim are not ablation rows")
        assert _instantiate(_resolve(path)).name == path.stem


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
class TestAblationGrid:
    def test_covers_rows_one_to_thirteen_exactly_once(self) -> None:
        """``MILESTONES.md`` section 6 has 13 rows; the grid must regenerate all of them."""
        rows = sorted(_resolve(p)["ablation_row"] for p in ROW_CONFIGS)
        assert rows == list(range(1, 14)), f"expected rows 1-13, got {rows}"

    def test_filenames_encode_their_row_number(self) -> None:
        for path in ROW_CONFIGS:
            declared = _resolve(path)["ablation_row"]
            assert path.name.startswith(f"row{declared:02d}_"), (
                f"{path.name} declares ablation_row={declared}; filename must agree"
            )

    def test_only_row_thirteen_uses_a_leaky_split(self) -> None:
        """Row 13 is deliberately wrong; any other row doing this inflates its score."""
        for path in ALL_CONFIGS:
            cfg = _instantiate(_resolve(path))
            if cfg.data.split_strategy == "random_chip":
                assert cfg.ablation_row == 13, (
                    f"{path.name} leaks spatially but is not row 13. "
                    "See docs/05 section 6.1."
                )

    def test_row_thirteen_is_the_leakage_row(self) -> None:
        cfg = _instantiate(_resolve(CONFIG_DIR / "row13_fusion_random_split.yaml"))
        assert cfg.data.split_strategy == "random_chip"

    def test_every_other_row_is_basin_blocked(self) -> None:
        for path in ROW_CONFIGS:
            cfg = _instantiate(_resolve(path))
            if cfg.ablation_row != 13:
                assert cfg.data.split_strategy == "leave_one_basin_out"

    @pytest.mark.parametrize(
        ("filename", "architecture"),
        [
            ("row01_otsu_global.yaml", "threshold_baseline"),
            ("row02_baseline_adaptive.yaml", "threshold_baseline"),
            ("row03_baseline_mndwi.yaml", "threshold_baseline"),
            ("row04_unet_sar_only.yaml", "unet_sar"),
            ("row05_unet_sar_dem.yaml", "unet_sar"),
            ("row06_unet_optical_only.yaml", "unet_optical"),
            ("row07_fusion_early.yaml", "fusion_early"),
            ("row08_fusion_late.yaml", "fusion_late"),
            ("row09_fusion_crossattn.yaml", "fusion_crossattn"),
        ],
    )
    def test_row_architectures(self, filename: str, architecture: str) -> None:
        assert _instantiate(_resolve(CONFIG_DIR / filename)).model.architecture == architecture


class TestScientificCoreRows:
    """Rows 10, 11 and 13 establish that you *understand* the model (``docs/09`` 5.1)."""

    def test_row10_removes_modality_dropout(self) -> None:
        cfg = _instantiate(_resolve(CONFIG_DIR / "row10_fusion_no_modality_dropout.yaml"))
        assert cfg.model.modality_dropout_p == 0.0

    def test_row10_differs_from_row09_only_in_dropout(self) -> None:
        """An ablation that changes two things measures neither."""
        nine = _resolve(CONFIG_DIR / "row09_fusion_crossattn.yaml")
        ten = _resolve(CONFIG_DIR / "row10_fusion_no_modality_dropout.yaml")
        differing = {
            section
            for section in SECTIONS
            if nine.get(section) != ten.get(section)
        }
        assert differing == {"model"}
        nine_model, ten_model = nine["model"], ten["model"]
        changed = {k for k in nine_model if nine_model[k] != ten_model.get(k)}
        assert changed == {"modality_dropout_p"}

    def test_row11_forces_sar_only_at_inference(self) -> None:
        """The operational number: optical is usually absent (``docs/02`` section 6)."""
        cfg = _instantiate(_resolve(CONFIG_DIR / "row11_fusion_eval_sar_only.yaml"))
        assert cfg.eval.force_sar_only is True

    def test_row11_keeps_training_identical_to_row09(self) -> None:
        """Row 11 trains nothing; it re-evaluates the row-9 checkpoint."""
        nine = _resolve(CONFIG_DIR / "row09_fusion_crossattn.yaml")
        eleven = _resolve(CONFIG_DIR / "row11_fusion_eval_sar_only.yaml")
        assert nine["train"] == eleven["train"]
        assert nine["loss"] == eleven["loss"]
        assert nine["model"] == eleven["model"]

    def test_row12_reduces_to_plain_bce(self) -> None:
        r"""``dice_weight=0`` and :math:`\gamma=0` collapse the compound loss to BCE."""
        cfg = _instantiate(_resolve(CONFIG_DIR / "row12_fusion_bce_loss.yaml"))
        assert cfg.loss.dice_weight == 0.0
        assert cfg.loss.focal_gamma == 0.0
        assert cfg.loss.focal_alpha == pytest.approx(0.5)

    def test_row12_keeps_dice_smooth_positive(self) -> None:
        """Switch Dice off by its WEIGHT, never by degrading smoothing into the trap."""
        cfg = _instantiate(_resolve(CONFIG_DIR / "row12_fusion_bce_loss.yaml"))
        assert cfg.loss.dice_smooth > 0.0


class TestReferencedFilenamesResolve:
    """Filenames cited elsewhere in the repo must exist (task: config alignment)."""

    def test_makefile_train_p1_config_exists(self) -> None:
        makefile = (CONFIG_DIR.parent.parent / "Makefile").read_text(encoding="utf-8")
        assert "configs/row09_fusion_crossattn.yaml" in makefile
        assert (CONFIG_DIR / "row09_fusion_crossattn.yaml").is_file()

    def test_train_module_docstring_config_exists(self) -> None:
        train_py = (
            CONFIG_DIR.parent / "src" / "floodfusion" / "train.py"
        ).read_text(encoding="utf-8")
        assert "configs/row09_fusion_crossattn.yaml" in train_py

    def test_no_stale_config_names_remain(self) -> None:
        """The pre-standardisation names must not survive anywhere."""
        stale = ["unet_fusion.yaml", "row09_fusion_reference.yaml", "fusion_bf16.yaml"]
        root = CONFIG_DIR.parent.parent
        for doc in list(root.glob("*.md")) + list((CONFIG_DIR.parent / "docs").glob("*.md")):
            text = doc.read_text(encoding="utf-8")
            for name in stale:
                assert name not in text, f"{doc.name} still references {name}"
