"""The channel order and configuration invariants.

These tests pass on a fresh clone — ``config.py`` is given complete — and they exist to
stop a later edit from silently permuting the physics. If you change the channel order,
you must change these tests, and that friction is deliberate.
"""

from __future__ import annotations

import dataclasses

import pytest

from floodfusion.config import (
    N_CHANNELS,
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    OPTICAL_SLICE,
    SAR_SLICE,
    TERRAIN_SLICE,
    Channel,
    EvalConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    NormalizationStats,
    TrainConfig,
    ValidityFlag,
)


class TestChannelOrder:
    """The normative channel stack of ``docs/05`` section 4.1."""

    def test_channel_count(self) -> None:
        assert len(Channel) == N_CHANNELS == 12

    def test_channels_are_contiguous_and_zero_based(self) -> None:
        assert [int(c) for c in Channel] == list(range(N_CHANNELS))

    def test_modality_groups_are_contiguous(self) -> None:
        """Groups must be sliceable, so the Dataset avoids a fancy-index copy."""
        assert SAR_SLICE == slice(0, 5)
        assert OPTICAL_SLICE == slice(5, 9)
        assert TERRAIN_SLICE == slice(9, 12)

    def test_group_sizes_partition_the_stack(self) -> None:
        assert N_SAR_CHANNELS + N_OPTICAL_CHANNELS + N_TERRAIN_CHANNELS == N_CHANNELS

    def test_sar_group_membership(self) -> None:
        sar = {Channel.VV_POST, Channel.VH_POST, Channel.VV_PRE, Channel.VH_PRE,
               Channel.DELTA_VV}
        assert {int(c) for c in sar} == set(range(*SAR_SLICE.indices(N_CHANNELS)))

    def test_optical_group_membership(self) -> None:
        optical = {Channel.B3_GREEN, Channel.B8_NIR, Channel.B11_SWIR1, Channel.MNDWI}
        assert {int(c) for c in optical} == set(range(*OPTICAL_SLICE.indices(N_CHANNELS)))

    def test_terrain_group_membership(self) -> None:
        terrain = {Channel.SLOPE, Channel.LOG1P_HAND, Channel.LOCAL_INCIDENCE}
        assert {int(c) for c in terrain} == set(range(*TERRAIN_SLICE.indices(N_CHANNELS)))


class TestValidityFlags:
    """Exclusion causes must remain separable for error analysis (``docs/09`` section 6)."""

    def test_valid_is_zero(self) -> None:
        assert int(ValidityFlag.VALID) == 0

    def test_flags_are_distinct_powers_of_two(self) -> None:
        flags = [int(f) for f in ValidityFlag if f is not ValidityFlag.VALID]
        assert all(f & (f - 1) == 0 for f in flags), "each flag must be a single bit"
        assert len(set(flags)) == len(flags), "flags must not collide"

    def test_flags_compose_without_collision(self) -> None:
        combined = int(ValidityFlag.SHADOW) | int(ValidityFlag.PERMANENT_WATER)
        assert combined & int(ValidityFlag.SHADOW)
        assert combined & int(ValidityFlag.PERMANENT_WATER)
        assert not combined & int(ValidityFlag.LAYOVER)


class TestConfigInvariants:
    def test_configs_are_frozen(self) -> None:
        """Mutating a config mid-run destroys reproducibility; the type system forbids it."""
        cfg = ExperimentConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.name = "mutated"  # type: ignore[misc]

    def test_dice_smooth_must_be_positive(self) -> None:
        """With ``smooth == 0`` an all-negative chip yields zero gradient (``docs/08`` 2.4)."""
        with pytest.raises(ValueError, match="strictly positive"):
            LossConfig(dice_smooth=0.0)

    def test_attention_head_dim_bounded_for_flash_path(self) -> None:
        """``head_dim > 128`` silently disables FlashAttention (``docs/07`` 4.3)."""
        with pytest.raises(ValueError, match="128"):
            ModelConfig(attention_head_dim=256)

    def test_fusion_stage_within_encoder_depth(self) -> None:
        with pytest.raises(ValueError, match="fusion_stage"):
            ModelConfig(fusion_stage=9)

    def test_modality_dropout_is_a_probability(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ModelConfig(modality_dropout_p=1.5)

    def test_normalisation_stats_match_sar_channel_count(self) -> None:
        with pytest.raises(ValueError, match="sar_mean_db"):
            NormalizationStats(sar_mean_db=(0.0, 0.0))

    def test_normalisation_std_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            NormalizationStats(sar_std_db=(1.0, 1.0, 0.0, 1.0, 1.0))


class TestDecisionThreshold:
    r"""The threshold follows from the cost ratio, not from 0.5 (``docs/00`` section 6.2)."""

    def test_default_matches_cost_ratio_three(self) -> None:
        assert EvalConfig(cost_ratio=3.0).decision_threshold == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(1.0, 0.5), (3.0, 0.25), (9.0, 0.1)],
    )
    def test_tau_star_formula(self, ratio: float, expected: float) -> None:
        r""":math:`\tau^* = 1/(1+r)`. Symmetric costs are the special case, not the default."""
        assert EvalConfig(cost_ratio=ratio).decision_threshold == pytest.approx(expected)

    def test_symmetric_costs_recover_one_half(self) -> None:
        """0.5 is only correct when a missed flood costs the same as a false alarm."""
        assert EvalConfig(cost_ratio=1.0).decision_threshold == pytest.approx(0.5)


class TestTrainConfig:
    def test_effective_batch_is_the_product(self) -> None:
        cfg = TrainConfig(micro_batch_size=4, grad_accum_steps=4)
        assert cfg.effective_batch_size == 16

    def test_amp_dtype_is_bfloat16_not_float16(self) -> None:
        """The soft-Dice reduction sums ~5e5 terms and overflows fp16 (``docs/08`` 5.1)."""
        assert TrainConfig().amp_dtype == "bfloat16"
