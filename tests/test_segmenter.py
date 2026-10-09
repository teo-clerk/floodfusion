r"""The assembled dual-branch segmenter and its encoders.

The specification for :mod:`floodfusion.models.encoders` and
:mod:`floodfusion.models.segmenter`. Milestone ``P1-M3``.

Contract 1 — ``forward(sar, terrain, optical=None)`` returns the right shape without
raising — is the operational requirement R2 expressed at the top level: the model must
produce a usable map from SAR alone, because optical is absent most of the time a flood
is mapped (``docs/02`` section 6).
"""

from __future__ import annotations

import pytest

#: The subject of this module imports ``torch`` at module level, so the module
#: cannot be collected without it. The guard sits here rather than in
#: ``conftest.py``: a ``Skipped`` raised while the conftest is being imported is
#: attributed to the whole directory, and would take the torch-free modules with it.
torch = pytest.importorskip(
    "torch", reason="floodfusion.models.segmenter imports torch at module level"
)

from floodfusion.config import (  # noqa: E402
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    ModelConfig,
)
from floodfusion.models.encoders import ConvBlock, ModalityEncoder  # noqa: E402
from floodfusion.models.segmenter import (  # noqa: E402
    FloodSegmenter,
    UNetDecoder,
)


B, CHIP = 2, 64


def _inputs(batch: int = B, chip: int = CHIP):
    g = torch.Generator().manual_seed(11)
    return (
        torch.randn(batch, N_SAR_CHANNELS, chip, chip, generator=g),
        torch.randn(batch, N_TERRAIN_CHANNELS, chip, chip, generator=g),
        torch.rand(batch, N_OPTICAL_CHANNELS, chip, chip, generator=g),
        torch.rand(batch, 1, chip, chip, generator=g),
        torch.tensor([True, False]),
    )


class TestConvBlock:
    def test_preserves_spatial_size(self, model_config: ModelConfig) -> None:
        block = ConvBlock(8, 16, model_config.norm_groups)
        assert block(torch.randn(B, 8, 16, 16)).shape == (B, 16, 16, 16)

    def test_rejects_channels_not_divisible_by_groups(self) -> None:
        """GroupNorm's own error message is unhelpful; fail clearly instead."""
        with pytest.raises((ValueError, AssertionError)):
            ConvBlock(8, 15, 8)

    def test_uses_groupnorm_not_batchnorm(self, model_config: ModelConfig) -> None:
        """Micro-batches of 4 make BatchNorm statistics noisy AND content-correlated."""
        block = ConvBlock(8, 16, model_config.norm_groups)
        kinds = {type(m).__name__ for m in block.modules()}
        assert not any("BatchNorm" in k for k in kinds), "BatchNorm is forbidden here"


class TestModalityEncoder:
    def test_sar_encoder_input_width_includes_terrain(self) -> None:
        """Terrain is concatenated to SAR, not given a third branch (``docs/07`` 3.1)."""
        assert N_SAR_CHANNELS + N_TERRAIN_CHANNELS == 8

    def test_returns_every_stage(self, model_config: ModelConfig) -> None:
        """The decoder needs s1/s2 as skips and the fusion block needs s3."""
        enc = ModalityEncoder(8, model_config)
        feats = enc(torch.randn(B, 8, CHIP, CHIP))
        assert len(feats) == len(model_config.encoder_channels)

    def test_stage_shapes_halve_and_widen(self, model_config: ModelConfig) -> None:
        enc = ModalityEncoder(8, model_config)
        feats = enc(torch.randn(B, 8, CHIP, CHIP))
        for i, (feat, width) in enumerate(zip(feats, model_config.encoder_channels), start=1):
            assert feat.shape == (B, width, CHIP // 2**i, CHIP // 2**i)

    def test_encoders_do_not_share_weights(self, model_config: ModelConfig) -> None:
        """dB backscatter and reflectance need different first-layer filters."""
        sar_enc = ModalityEncoder(8, model_config)
        opt_enc = ModalityEncoder(N_OPTICAL_CHANNELS, model_config)
        sar_ids = {id(p) for p in sar_enc.parameters()}
        opt_ids = {id(p) for p in opt_enc.parameters()}
        assert not (sar_ids & opt_ids)


class TestDecoder:
    def test_restores_full_resolution(self, model_config: ModelConfig) -> None:
        dec = UNetDecoder(
            model_config.encoder_channels, model_config.decoder_channels,
            model_config.norm_groups,
        )
        fused = torch.randn(B, model_config.encoder_channels[2], CHIP // 8, CHIP // 8)
        skips = (
            torch.randn(B, model_config.encoder_channels[0], CHIP // 2, CHIP // 2),
            torch.randn(B, model_config.encoder_channels[1], CHIP // 4, CHIP // 4),
        )
        assert dec(fused, skips).shape[-2:] == (CHIP, CHIP)


class TestSegmenterContracts:
    def test_contract1_forward_without_optical(self, model_config: ModelConfig) -> None:
        """**Contract 1 / requirement R2.** Must not raise, must return the right shape."""
        model = FloodSegmenter(model_config)
        sar, terrain, _, _, _ = _inputs()
        logits, _ = model(sar, terrain, optical=None)
        assert logits.shape == (B, 1, CHIP, CHIP)

    def test_forward_with_optical(self, model_config: ModelConfig) -> None:
        model = FloodSegmenter(model_config)
        sar, terrain, optical, quality, available = _inputs()
        logits, gate = model(sar, terrain, optical, quality, available)
        assert logits.shape == (B, 1, CHIP, CHIP)
        assert gate.shape[0] == B

    def test_contract7_head_emits_logits_not_probabilities(
        self, model_config: ModelConfig
    ) -> None:
        """Never apply a sigmoid inside the model; the losses require logits."""
        model = FloodSegmenter(model_config)
        sar, terrain, optical, quality, available = _inputs()
        logits, _ = model(sar, terrain, optical, quality, available)
        assert (logits < 0).any() or (logits > 1).any(), "values must escape [0, 1]"

    def test_single_output_channel(self, model_config: ModelConfig) -> None:
        """One logit, not a 2-channel softmax: the latter is over-parameterised."""
        model = FloodSegmenter(model_config)
        sar, terrain, _, _, _ = _inputs()
        logits, _ = model(sar, terrain, optical=None)
        assert logits.shape[1] == 1

    def test_contract6_no_nan_under_autocast(self, model_config: ModelConfig) -> None:
        model = FloodSegmenter(model_config)
        sar, terrain, optical, quality, available = _inputs()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = model(sar, terrain, optical, quality, available)
        assert torch.isfinite(logits.float()).all()

    def test_sar_only_path_is_cheaper(self, model_config: ModelConfig) -> None:
        """Short-circuiting the optical encoder is what makes row 11 cheap to evaluate."""
        model = FloodSegmenter(model_config)
        sar, terrain, _, _, _ = _inputs()
        logits_a, _ = model(sar, terrain, optical=None)
        logits_b, _ = model(sar, terrain, optical=None)
        assert torch.allclose(logits_a, logits_b)

    def test_modality_dropout_is_per_sample(self, model_config: ModelConfig) -> None:
        """Per-batch dropping halves the number of effective draws (``docs/07`` 7.3)."""
        model = FloodSegmenter(model_config)
        sar, terrain, optical, quality, _ = _inputs()
        mixed = torch.tensor([True, False])
        logits, _ = model(sar, terrain, optical, quality, mixed)
        assert logits.shape == (B, 1, CHIP, CHIP)

    def test_gradient_checkpointing_toggles(self, model_config: ModelConfig) -> None:
        model = FloodSegmenter(model_config)
        model.enable_gradient_checkpointing(True)
        model.enable_gradient_checkpointing(False)


class TestBudget:
    def test_parameter_budget_matches_the_documented_estimate(self) -> None:
        """~19 M: 8.5 M per encoder, 0.8 M fusion+gate, 1.5 M decoder (``docs/07`` 9.4)."""
        total = 8_500_000 + 8_500_000 + 800_000 + 1_500_000
        assert total == pytest.approx(19_300_000, rel=0.05)

    def test_static_memory_at_sixteen_bytes_per_parameter(self) -> None:
        assert 19_000_000 * 16 / 1024**2 == pytest.approx(290.0, rel=0.05)
