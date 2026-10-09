"""Frozen configuration for Project 01.

Every configuration object here is an immutable ``@dataclass(frozen=True)``. Mutating a
config after a run has started is the single easiest way to make a result
irreproducible, so the type system forbids it: use :func:`dataclasses.replace` to derive
a modified copy.

The **channel order defined in this module is normative**. It is asserted by
``tests/test_dataset.py`` and relied upon by every model in ``models/``. Changing it
without changing the tests will produce a model that trains happily on permuted physics.

References
----------
``docs/05_dataset_construction_and_splits.md`` section 4.1 — the channel stack.
``docs/05_dataset_construction_and_splits.md`` section 7   — normalisation policy.
``docs/03_dem_hand_and_terrain_effects.md``   section 3    — the analysis grid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Final


# ---------------------------------------------------------------------------
# Channel order — NORMATIVE. See docs/05 section 4.1.
# ---------------------------------------------------------------------------
class Channel(IntEnum):
    """Index of each channel in the stacked ``(C, H, W)`` chip array.

    The three modality groups are contiguous on purpose: the ``Dataset`` slices them
    with ``[SAR_SLICE]``, ``[OPTICAL_SLICE]`` and ``[TERRAIN_SLICE]`` rather than by
    fancy indexing, which keeps the arrays contiguous and avoids a copy.
    """

    # --- SAR group (always present) ---------------------------------------
    VV_POST = 0
    VH_POST = 1
    VV_PRE = 2
    VH_PRE = 3
    DELTA_VV = 4
    # --- Optical group (intermittent) -------------------------------------
    B3_GREEN = 5
    B8_NIR = 6
    B11_SWIR1 = 7
    MNDWI = 8
    # --- Terrain group (always present) -----------------------------------
    SLOPE = 9
    LOG1P_HAND = 10
    LOCAL_INCIDENCE = 11


N_CHANNELS: Final[int] = 12

SAR_SLICE: Final[slice] = slice(0, 5)
OPTICAL_SLICE: Final[slice] = slice(5, 9)
TERRAIN_SLICE: Final[slice] = slice(9, 12)

N_SAR_CHANNELS: Final[int] = 5
N_OPTICAL_CHANNELS: Final[int] = 4
N_TERRAIN_CHANNELS: Final[int] = 3


class ValidityFlag(IntEnum):
    """Bit flags packed into the per-pixel exclusion mask.

    Stored as a single ``uint8`` band rather than five boolean bands: it is five times
    smaller on disk and lets you attribute an exclusion to its cause during error
    analysis (``docs/09`` section 6), which a collapsed boolean mask cannot.

    A pixel is valid for loss and metric iff its flag byte is exactly ``0``.
    """

    VALID = 0
    NODATA = 1 << 0
    LAYOVER = 1 << 1
    SHADOW = 1 << 2
    PERMANENT_WATER = 1 << 3
    OUTSIDE_AOI = 1 << 4
    STEEP_SLOPE = 1 << 5


# ---------------------------------------------------------------------------
# Sensor and grid constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorConstants:
    """Physical constants of the Sentinel-1 C-band instrument.

    Used by the geometry code in :mod:`floodfusion.masks` and by the analytic checks in
    ``tests/``. See ``docs/01_sar_hydrology_and_water_backscatter.md`` section 1.
    """

    carrier_frequency_hz: float = 5.405e9
    wavelength_m: float = 0.05551
    incidence_min_deg: float = 29.1
    incidence_max_deg: float = 46.0
    equivalent_number_of_looks: float = 4.4
    #: Noise-equivalent sigma-nought, dB. Below this you measure the receiver, not the
    #: surface (docs/01 section 3.3).
    nesz_db: float = -22.0


@dataclass(frozen=True)
class GridConfig:
    """The single analysis grid. See ``docs/03`` section 3.

    ``origin_snap_m`` makes the grid a deterministic function of the AOI alone, so two
    layers processed independently land on exactly the same grid with no resampling
    phase difference. Do not set it to ``0``.
    """

    pixel_size_m: float = 10.0
    origin_snap_m: float = 10.0
    #: ``None`` means "derive the UTM zone from the AOI centroid" (docs/03 section 3.1).
    epsg: int | None = None
    nodata_int16: int = -32768
    #: dB values are stored as ``round(db * db_scale)`` in int16 (docs/04 section 8.1).
    db_scale: float = 100.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizationStats:
    """Frozen dataset-level normalisation statistics.

    **These are computed on the training split only** and then frozen here. Computing
    them over train+val+test leaks test information into training
    (``docs/05`` section 6.5 point 1).

    **Never normalise per chip.** A flooded chip is dark, and that darkness is the
    signal; per-chip standardisation deletes it (``docs/05`` section 7.1).

    Attributes
    ----------
    sar_mean_db, sar_std_db
        Per-channel z-score statistics for the five SAR channels, in dB.
    optical_scale
        Sentinel-2 ``QUANTIFICATION_VALUE``. Read it from the product metadata; do not
        hard-code, because it changes with processing baseline.
    optical_offset
        Sentinel-2 ``BOA_ADD_OFFSET`` (typically ``-1000`` from baseline 04.00).
        Ignoring it shifts every reflectance by 0.1, which is larger than the water
        signal itself (``docs/05`` section 7.2).
    hand_log_scale
        Divisor applied after ``log1p(HAND)`` to bring the channel to roughly ``[0, 1]``.
    """

    sar_mean_db: tuple[float, ...] = (-12.5, -18.2, -12.3, -18.0, 0.0)
    sar_std_db: tuple[float, ...] = (4.8, 5.1, 4.8, 5.1, 3.1)
    optical_scale: float = 10000.0
    optical_offset: float = -1000.0
    hand_log_scale: float = 3.932  # log1p(50)
    slope_scale_deg: float = 90.0
    incidence_scale_deg: float = 90.0

    def __post_init__(self) -> None:
        if len(self.sar_mean_db) != N_SAR_CHANNELS:
            raise ValueError(
                f"sar_mean_db must have {N_SAR_CHANNELS} entries, got {len(self.sar_mean_db)}"
            )
        if len(self.sar_std_db) != N_SAR_CHANNELS:
            raise ValueError(
                f"sar_std_db must have {N_SAR_CHANNELS} entries, got {len(self.sar_std_db)}"
            )
        if any(s <= 0.0 for s in self.sar_std_db):
            raise ValueError("sar_std_db entries must be strictly positive")
        if self.optical_scale <= 0.0:
            raise ValueError("optical_scale must be strictly positive")


@dataclass(frozen=True)
class DataConfig:
    """Chipping, sampling and split configuration. See ``docs/05`` sections 6 and 8."""

    chip_root: Path = Path("chips")
    manifest_name: str = "manifest.parquet"
    split_file_name: str = "splits.json"
    #: Train-only normalisation statistics of that split (one per cross-validation fold).
    norm_stats_file: str = "norm_stats.json"

    chip_size: int = 256
    train_stride: int = 128  # 50 % overlap; acceptable WITHIN a split only
    eval_stride: int = 256  # no overlap; every pixel scored exactly once

    #: Drop chips whose valid fraction falls below this.
    min_valid_fraction: float = 0.5
    #: A chip counts as "positive" for sampling if its flood fraction exceeds this.
    positive_fraction_threshold: float = 0.005
    #: Negative:positive chip ratio after subsampling. Changes the prior — the
    #: correction of ``docs/05`` section 8.3 is then MANDATORY before thresholding.
    negative_to_positive_ratio: float = 3.0

    #: Buffer applied at basin boundaries. MUST exceed the empirical variogram range
    #: measured in ``docs/05`` section 6.2. Do not guess this.
    basin_buffer_m: float = 1000.0

    #: Cross-validation protocol. ``leave_one_basin_out`` is mandatory for every
    #: reported number; ``random_chip`` is **deliberately leaky** and exists solely for
    #: ablation row 13, which quantifies the spatial-leakage illusion
    #: (``docs/05`` section 6, ``docs/09`` section 5.3).
    split_strategy: str = "leave_one_basin_out"

    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    #: Where chip pixels come from. ``raster`` reads the stacked GeoTIFFs and fails loudly
    #: if one is missing; ``synthetic`` renders every chip in memory from the physics
    #: simulator (offline smoke tests); ``auto`` reads the raster when the file exists and
    #: falls back to the in-memory simulator only when it does not, with a warning.
    #: ``configs/base.yaml`` pins ``raster`` so no reported run can silently train on
    #: simulated pixels; the dataclass default exists for unit tests and notebooks.
    chip_source: str = "auto"

    CHIP_SOURCES: ClassVar[frozenset[str]] = frozenset({"auto", "raster", "synthetic"})

    def __post_init__(self) -> None:
        allowed = {"leave_one_basin_out", "random_chip", "published"}
        if self.split_strategy not in allowed:
            raise ValueError(
                f"split_strategy must be one of {sorted(allowed)}, "
                f"got {self.split_strategy!r}. Spatially blocked validation is not "
                "optional here — see docs/05 section 6."
            )
        if self.basin_buffer_m < 0.0:
            raise ValueError("basin_buffer_m must be non-negative")
        if self.chip_source not in self.CHIP_SOURCES:
            raise ValueError(
                f"chip_source must be one of {sorted(self.CHIP_SOURCES)}, "
                f"got {self.chip_source!r}"
            )
        if self.chip_size % 16 != 0:
            raise ValueError(
                "chip_size must be a multiple of 16: the encoder halves resolution four "
                f"times; got {self.chip_size}"
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration. See ``docs/07_fusion_architectures.md``.

    ``fusion_stage`` selects the encoder stage at which cross-attention runs. Stage 3
    (``32 x 32`` for a 256-pixel chip) is the default because the naive attention matrix
    is 134 MB there against 2.15 GB at stage 2 — see ``docs/07`` section 6 for the
    arithmetic, which you should reproduce for your own batch size.
    """

    #: Which architecture this row instantiates. The ablation grid of
    #: ``MILESTONES.md`` section 6 is expressed almost entirely through this field.
    architecture: str = "fusion_crossattn"

    #: Whether the terrain channels are concatenated to the SAR encoder input.
    #: Rows 4 and 5 differ only in this (``docs/07`` section 3.1).
    use_terrain: bool = True

    encoder_channels: tuple[int, ...] = (64, 128, 256, 512)
    fusion_stage: int = 3  # 1-indexed; s3 == 32x32 for a 256 chip
    attention_heads: int = 8
    attention_head_dim: int = 32  # must satisfy head_dim <= 128 for the Flash path
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32)
    norm_groups: int = 8  # GroupNorm, not BatchNorm — docs/07 section 3.3
    dropout: float = 0.1

    #: Probability of replacing the optical branch with the learned missing-modality
    #: embedding during training. The principled value matches the DEPLOYMENT frequency
    #: of missing optical measured in ``docs/02`` section 6.3 — not 0.5 by default.
    modality_dropout_p: float = 0.5

    #: Initial bias of the gate. Negative so the model starts near SAR-only and must
    #: earn its use of optical (``docs/07`` section 8.3).
    gate_bias_init: float = -2.0

    #: Architectures the ablation grid uses. ``threshold_baseline`` is not a network at
    #: all — rows 1--3 run on CPU through :mod:`floodfusion.baselines`.
    ARCHITECTURES: ClassVar[frozenset[str]] = frozenset(
        {
            "threshold_baseline",  # rows 1-3
            "unet_sar",  # rows 4, 5
            "unet_optical",  # row 6
            "fusion_early",  # row 7
            "fusion_late",  # row 8
            "fusion_crossattn",  # rows 9-13 (the reference design)
        }
    )

    def __post_init__(self) -> None:
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {sorted(self.ARCHITECTURES)}, "
                f"got {self.architecture!r}"
            )
        if not 1 <= self.fusion_stage <= len(self.encoder_channels):
            raise ValueError(
                f"fusion_stage must be in [1, {len(self.encoder_channels)}], "
                f"got {self.fusion_stage}"
            )
        if self.attention_head_dim > 128:
            raise ValueError(
                "attention_head_dim > 128 disables the FlashAttention path "
                "(docs/07 section 4.3)"
            )
        if not 0.0 <= self.modality_dropout_p <= 1.0:
            raise ValueError("modality_dropout_p must lie in [0, 1]")


# ---------------------------------------------------------------------------
# Loss, training, evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LossConfig:
    """Compound focal + soft-Dice loss. See ``docs/08`` section 2.

    ``dice_smooth`` appears in **both numerator and denominator**. With smoothing in the
    denominator only, an all-negative chip produces exactly zero gradient — and most
    chips are all-negative (``docs/08`` section 2.4). This is not a tuning parameter;
    it is a correctness condition.
    """

    focal_weight: float = 1.0
    dice_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    dice_smooth: float = 1.0

    def __post_init__(self) -> None:
        if self.dice_smooth <= 0.0:
            raise ValueError(
                "dice_smooth must be strictly positive; see docs/08 section 2.4 "
                "for the all-negative-chip trap"
            )
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser, schedule and memory policy. See ``docs/08`` sections 5 and 7."""

    epochs: int = 60
    micro_batch_size: int = 4
    grad_accum_steps: int = 4  # effective batch = 4 * 4 = 16
    learning_rate: float = 1.0e-4
    min_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-2  # excluded from norm and bias parameters
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    #: Epochs without a new best validation macro IoU before stopping; 0 disables.
    early_stopping_patience: int = 0

    #: bf16, never fp16: the soft-Dice reduction sums ~5e5 terms and overflows fp16's
    #: 5-bit exponent (``docs/08`` section 5.1).
    amp_dtype: str = "bfloat16"
    channels_last: bool = True
    grad_checkpointing: bool = False  # needed at 512, not at 256
    cudnn_benchmark: bool = True  # only valid because chip shapes are FIXED
    #: ``torch.use_deterministic_algorithms(True, warn_only=True)`` (docs/08 7.2). Measured on the
    #: RTX 4060 Laptop: 2.9x slower per chip for the reference model. The docs' defensible
    #: position is to seed everything, accept residual non-determinism and run three seeds.
    deterministic: bool = False

    seed: int = 42

    def __post_init__(self) -> None:
        # bf16 is mandated, not preferred. The soft-Dice reduction sums B*H*W terms --
        # 524 288 at B=8, 256^2 -- and a sum of order 1e5 OVERFLOWS fp16's 5-bit
        # exponent (max 65 504). A GradScaler does not help: it addresses vanishing
        # gradients in the backward pass, not an overflow in the forward reduction.
        # See docs/08 section 5.1.
        allowed = {"bfloat16", "float32"}
        if self.amp_dtype not in allowed:
            raise ValueError(
                f"amp_dtype must be one of {sorted(allowed)}, got {self.amp_dtype!r}. "
                "float16 overflows the soft-Dice reduction — see docs/08 section 5.1."
            )
        if self.micro_batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("micro_batch_size and grad_accum_steps must be >= 1")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 (0 disables)")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


@dataclass(frozen=True)
class BaselineConfig:
    """Parameters of the non-learned physics baseline. See ``docs/06`` sections 4--5.

    Every value here is tuned on the **validation** split and then frozen. Tuning on
    test makes the test set a validation set (``docs/05`` section 6.5). A baseline you
    cannot regenerate is not a baseline, so these belong in the config, not in a
    notebook.
    """

    #: ``global_otsu`` is ablation row 1 — the floor, expected to fail badly because a
    #: 2 % water scene is not bimodal (``docs/06`` section 3.3). ``tile_adaptive`` is
    #: the operational reference of row 2.
    threshold_mode: str = "tile_adaptive"

    #: Row 1 runs with every exclusion disabled. Rows 2+ enable them, and toggling them
    #: one at a time produces the per-mask attribution table of ``docs/06`` section 6,
    #: which ``P1-M2.4`` requires.
    apply_terrain_masks: bool = True
    apply_permanent_water_mask: bool = True

    tile_size: int = 200
    min_tile_valid_fraction: float = 0.5
    #: Ashman's D acceptance threshold; > 2 indicates two separable populations.
    ashman_min: float = 2.0
    #: Neither mixture component may be negligible — water fraction in [10 %, 90 %].
    min_component_weight: float = 0.10
    #: The dark mode must actually be dark enough to be water, guarding against tiles
    #: that are bimodal for an irrelevant reason (forest against bare field).
    water_mode_max_db: float = -14.0

    #: Change-detection threshold in dB. Derived from the speckle statistics as
    #: k * sigma_delta_dB(L_eff) with k ~ 2--3, NOT chosen as a round number
    #: (``docs/01`` section 10.2). At L = 4.4, sigma_delta = 3.07 dB, so 2 sigma ~ 6.1.
    delta_min_db: float = 6.0
    tau_mndwi: float = 0.0
    slope_max_deg: float = 10.0
    hand_max_m: float = 15.0

    #: Conjunction of amplitude AND change. The OR variant is a required comparison in
    #: ``docs/06`` section 5.2 — measure it, do not assume it.
    use_change_term: bool = True
    #: Whether a valid optical observation overrides the SAR decision.
    use_optical_override: bool = True
    #: Speckle filtering is unavoidable for a per-pixel threshold at L = 4.4.
    speckle_filter: str = "refined_lee"
    speckle_window: int = 7
    #: Region growing (docs/06 section 4.2 step 7): from the SAR decision, grow into
    #: connected pixels within this many dB above the threshold surface that also show
    #: at least half the required darkening. Recovers boundary pixels lost to speckle
    #: without admitting permanently dark surfaces. Not applied to row 1.
    region_grow_margin_db: float = 1.5

    THRESHOLD_MODES: ClassVar[frozenset[str]] = frozenset({"global_otsu", "tile_adaptive"})

    def __post_init__(self) -> None:
        if self.threshold_mode not in self.THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {sorted(self.THRESHOLD_MODES)}, "
                f"got {self.threshold_mode!r}"
            )
        if self.delta_min_db <= 0.0:
            raise ValueError(
                "delta_min_db is a magnitude and must be positive; the rule tests "
                "(post - pre) < -delta_min_db"
            )
        if not 0.0 < self.min_component_weight < 0.5:
            raise ValueError("min_component_weight must lie in (0, 0.5)")
        if self.region_grow_margin_db < 0.0:
            raise ValueError("region_grow_margin_db must be non-negative")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol. See ``docs/06`` and ``docs/09``."""

    #: Force the optical branch to the missing-modality embedding at INFERENCE, whatever
    #: the data offers. This is ablation row 11 — the operational number, because
    #: optical is usually absent (``docs/02`` section 6). It changes evaluation only;
    #: the model is the row-9 checkpoint, unmodified.
    force_sar_only: bool = False

    #: Cost ratio c_FN / c_FP from ``docs/00`` section 6. The decision threshold
    #: FOLLOWS from this; it is not chosen independently.
    cost_ratio: float = 3.0
    n_bootstrap: int = 10_000
    bootstrap_alpha: float = 0.05
    #: Fixed thresholds at which confusion matrices are accumulated. Accumulating
    #: matrices rather than storing probability maps is what keeps evaluation inside
    #: the VRAM budget (``docs/08`` section 7.3).
    pr_thresholds: tuple[float, ...] = tuple(i / 200.0 for i in range(1, 200))

    @property
    def decision_threshold(self) -> float:
        r"""Cost-optimal threshold :math:`\tau^* = 1 / (1 + r)`.

        Derived in ``docs/00`` section 6.2. Applying it to **uncalibrated** scores is
        numerology — fit a temperature on validation first (``docs/09`` section 4).
        """
        return 1.0 / (1.0 + self.cost_ratio)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration. One frozen instance per ablation row."""

    name: str = "p01_fusion_reference"
    ablation_row: int = 9
    output_dir: Path = Path("runs")

    sensor: SensorConstants = field(default_factory=SensorConstants)
    grid: GridConfig = field(default_factory=GridConfig)
    norm: NormalizationStats = field(default_factory=NormalizationStats)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.ablation_row <= 13:
            raise ValueError(
                f"ablation_row must be in [0, 13] to map onto MILESTONES.md section 6; "
                f"got {self.ablation_row}. Use 0 for a configuration that is not a row."
            )
        # Row 13 is the ONLY configuration permitted to use a leaky split, and it must
        # declare itself as such. Guarding here means a stray `random_chip` in any other
        # row is a construction-time error rather than an inflated headline number.
        if self.data.split_strategy == "random_chip" and self.ablation_row != 13:
            raise ValueError(
                "split_strategy='random_chip' leaks spatially and is reserved for "
                f"ablation row 13; this config declares row {self.ablation_row}. "
                "See docs/05 section 6.1 and docs/09 section 5.3."
            )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
#: YAML section name -> dataclass. The schema of every file under ``configs/``.
SECTION_TYPES: Final[dict[str, type]] = {
    "sensor": SensorConstants,
    "grid": GridConfig,
    "norm": NormalizationStats,
    "data": DataConfig,
    "model": ModelConfig,
    "baseline": BaselineConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}
#: Top-level scalar keys permitted in a config file.
SCALAR_KEYS: Final[frozenset[str]] = frozenset({"name", "ablation_row", "output_dir"})
#: Fields stored as ``Path`` objects (YAML carries them as strings).
_PATH_FIELDS: Final[dict[str, frozenset[str]]] = {"data": frozenset({"chip_root"})}
_MAX_EXTENDS_DEPTH: Final[int] = 8


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml_chain(path: Path, depth: int = 0) -> dict[str, Any]:
    """Resolve an ``extends:`` chain (paths relative to the extending file)."""
    import yaml

    if depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(f"extends chain deeper than {_MAX_EXTENDS_DEPTH} at {path}; cycle?")
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml_chain(path.parent / str(parent), depth + 1), raw)


def _apply_override(raw: dict[str, Any], assignment: str) -> dict[str, Any]:
    """Apply one ``section.field=value`` override; the value is parsed as YAML."""
    import yaml

    if "=" not in assignment:
        raise ValueError(f"override {assignment!r} must look like section.field=value")
    dotted, _, text = assignment.partition("=")
    keys = [k.strip() for k in dotted.strip().split(".") if k.strip()]
    if not keys:
        raise ValueError(f"override {assignment!r} names no key")
    patch: Any = yaml.safe_load(text)
    for key in reversed(keys):
        patch = {key: patch}
    return _deep_merge(raw, patch)


def resolve_config_dict(
    path: Path | str,
    *,
    overlays: Sequence[Path | str] = (),
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve ``extends``, then overlays (partial YAML files), then ``a.b=v`` overrides.

    The returned mapping is exactly what :func:`config_from_dict` instantiates and what a
    run directory snapshots as ``config_resolved.yaml`` (``P1-M6.3``).
    """
    import yaml

    raw = _read_yaml_chain(Path(path))
    for overlay in overlays:
        overlay_path = Path(overlay)
        if not overlay_path.is_file():
            raise FileNotFoundError(f"overlay not found: {overlay_path}")
        patch = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        if not isinstance(patch, dict):
            raise ValueError(f"{overlay_path}: an overlay must be a mapping")
        if "extends" in patch:
            raise ValueError(f"{overlay_path}: overlays cannot use 'extends'")
        raw = _deep_merge(raw, patch)
    for assignment in overrides:
        raw = _apply_override(raw, assignment)
    return raw


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Instantiate an :class:`ExperimentConfig`, rejecting **every** unknown key at once."""
    problems: list[str] = []
    unknown_top = sorted(set(raw) - SCALAR_KEYS - set(SECTION_TYPES))
    if unknown_top:
        problems.append(f"unknown top-level keys {unknown_top}")
    for section, cls in SECTION_TYPES.items():
        values = raw.get(section, {}) or {}
        if not isinstance(values, Mapping):
            problems.append(f"[{section}] must be a mapping")
            continue
        known = {f.name for f in fields(cls)}
        extra = sorted(set(values) - known)
        if extra:
            problems.append(f"[{section}] unknown fields {extra}")
    if problems:
        raise ValueError("invalid configuration: " + "; ".join(problems))

    sections: dict[str, Any] = {}
    for section, cls in SECTION_TYPES.items():
        values = {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in dict(raw.get(section, {}) or {}).items()
        }
        for key in _PATH_FIELDS.get(section, frozenset()):
            if key in values and values[key] is not None:
                values[key] = Path(values[key])
        sections[section] = cls(**values)
    scalars = {k: raw[k] for k in SCALAR_KEYS if k in raw}
    if "output_dir" in scalars:
        scalars["output_dir"] = Path(scalars["output_dir"])
    return ExperimentConfig(**scalars, **sections)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain, YAML/JSON-serialisable view of a config (tuples -> lists, Path -> str)."""

    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple | list):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return value

    return {k: plain(v) for k, v in asdict(cfg).items()}


def config_sha256(raw: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a resolved config mapping.

    Canonical (sorted keys, no whitespace) so two semantically identical resolutions —
    e.g. one with and one without a redundant overlay — hash identically.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(
    path: Path | str,
    *,
    overlays: Sequence[Path | str] = (),
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML file under ``configs/``. Its ``extends:`` chain is resolved
        relative to the file, then deep-merged (child wins).
    overlays
        Partial YAML mappings deep-merged on top, in order — e.g.
        ``configs/overlays/synthetic_demo.yaml``. Overlays may not ``extend``.
    overrides
        ``section.field=value`` strings applied last; values are parsed as YAML, so
        ``train.epochs=3`` yields an ``int``.

    Returns
    -------
    ExperimentConfig
        A fully populated, frozen configuration. Use :func:`resolve_config_dict` and
        :func:`config_sha256` to record exactly what was loaded.

    Raises
    ------
    FileNotFoundError
        If ``path`` or an overlay does not exist.
    ValueError
        If the YAML contains keys that do not correspond to config fields (all of them are
        listed, not just the first), or if any ``__post_init__`` validation fails.

    Notes
    -----
    Unknown keys are an **error**, not a warning. A silently ignored typo in a config
    key is a reproducibility failure: the run proceeds with a default you did not
    intend and the log records a value you never used.
    """
    return config_from_dict(resolve_config_dict(path, overlays=overlays, overrides=overrides))
