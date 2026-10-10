"""Every ablation-grid architecture, and the per-sample routing inside the fusion model."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    ModelConfig,
)
from floodfusion.models.segmenter import FloodSegmenter
from floodfusion.models.zoo import (
    EarlyFusionSegmenter,
    LateFusionSegmenter,
    UNetSegmenter,
    build_model,
    count_parameters,
)

B, S = 3, 32
SMALL = dict(encoder_channels=(16, 32, 64, 128), decoder_channels=(64, 32, 16, 8),
             attention_heads=4, attention_head_dim=16, norm_groups=4)


def _cfg(architecture: str, **kw) -> ModelConfig:
    return ModelConfig(architecture=architecture, **{**SMALL, **kw})


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        sar=torch.randn(B, N_SAR_CHANNELS, S, S, generator=g),
        terrain=torch.rand(B, N_TERRAIN_CHANNELS, S, S, generator=g),
        optical=torch.rand(B, N_OPTICAL_CHANNELS, S, S, generator=g),
        optical_quality=torch.rand(B, 1, S, S, generator=g),
        optical_available=torch.tensor([True, False, True]),
    )


@pytest.mark.parametrize(
    "architecture", ["unet_sar", "fusion_early", "fusion_late", "fusion_crossattn"]
)
@pytest.mark.parametrize("with_optical", [True, False])
def test_every_architecture_emits_one_logit_per_pixel(architecture: str, with_optical: bool) -> None:
    model = build_model(_cfg(architecture))
    batch = _batch()
    if not with_optical:
        batch.update(optical=None, optical_quality=None, optical_available=None)
    logits, _ = model(**batch)
    assert logits.shape == (B, 1, S, S)
    assert torch.isfinite(logits).all()


def test_unet_optical_requires_optical() -> None:
    model = build_model(_cfg("unet_optical"))
    batch = _batch()
    assert model(**batch)[0].shape == (B, 1, S, S)
    with pytest.raises(ValueError, match="row 6"):
        model(batch["sar"], batch["terrain"], optical=None)


def test_threshold_baseline_is_not_a_network() -> None:
    with pytest.raises(ValueError, match="baseline_runner"):
        build_model(_cfg("threshold_baseline"))


def test_sar_only_unet_ignores_terrain_when_asked() -> None:
    model = build_model(_cfg("unet_sar", use_terrain=False))
    assert isinstance(model, UNetSegmenter) and model.in_channels == N_SAR_CHANNELS


def test_unsupported_fusion_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="fusion_stage"):
        FloodSegmenter(_cfg("fusion_crossattn", fusion_stage=2))


def test_shared_backbone_sizes_are_comparable() -> None:
    """Rows differ in fusion, not capacity: single-branch models share one backbone size."""
    sar = count_parameters(build_model(_cfg("unet_sar")))
    early = count_parameters(build_model(_cfg("fusion_early")))
    late = count_parameters(build_model(_cfg("fusion_late")))
    assert abs(early - sar) / sar < 0.01
    assert late == pytest.approx(2 * sar, rel=0.02)


class TestEarlyFusion:
    def test_absent_optical_is_the_constant_fill(self) -> None:
        model = EarlyFusionSegmenter(_cfg("fusion_early")).eval()
        batch = _batch()
        logits, _ = model(**batch)
        zeroed = dict(batch, optical=torch.zeros_like(batch["optical"]))
        ref, _ = model(**zeroed)
        assert torch.allclose(logits[1], ref[1], atol=1e-6), "sample 1 is unavailable -> zeros"
        assert not torch.allclose(logits[0], ref[0]), "sample 0 is available -> real optical"


class TestLateFusion:
    def test_absent_optical_reduces_exactly_to_the_sar_branch(self) -> None:
        model = LateFusionSegmenter(_cfg("fusion_late")).eval()
        batch = _batch()
        logits, gate = model(**batch)
        sar_only = model.sar_net(torch.cat([batch["sar"], batch["terrain"]], 1))
        assert torch.equal(logits[1], sar_only[1])
        assert torch.equal(gate[1], torch.zeros_like(gate[1]))
        assert not torch.equal(logits[0], sar_only[0])


class TestFusionRouting:
    def test_mixed_batch_matches_per_sample_forwards(self) -> None:
        """GroupNorm is per-sample, so running the optical encoder on a subset is exact."""
        model = FloodSegmenter(_cfg("fusion_crossattn")).eval()
        batch = _batch()
        mixed, _ = model(**batch)
        for i in range(B):
            single = {k: (v[i:i + 1] if v is not None else None) for k, v in batch.items()}
            alone, _ = model(**single)
            assert torch.allclose(mixed[i:i + 1], alone, atol=1e-5)

    def test_gradients_follow_availability_per_sample(self) -> None:
        model = FloodSegmenter(_cfg("fusion_crossattn"))
        batch = _batch()
        batch["optical"].requires_grad_(True)
        with torch.no_grad():
            model.fusion.gate.out.bias.fill_(4.0)  # open the gate so attention matters
        logits, _ = model(**batch)
        logits.sum().backward()
        grad = batch["optical"].grad
        assert grad[0].abs().sum() > 0 and grad[2].abs().sum() > 0
        assert grad[1].abs().sum() == 0, "unavailable sample: no gradient into its optical input"
        assert model.missing_optical.embedding.grad.abs().sum() > 0

    def test_all_absent_skips_the_optical_encoder(self) -> None:
        model = FloodSegmenter(_cfg("fusion_crossattn"))
        batch = _batch()
        batch["optical_available"] = torch.zeros(B, dtype=torch.bool)
        logits, _ = model(**batch)
        logits.sum().backward()
        assert all(p.grad is None for p in model.optical_encoder.parameters())
        a, _ = model(**batch)
        b, _ = model(batch["sar"], batch["terrain"], optical=None)
        assert torch.allclose(a, b, atol=1e-6), "all-False availability == optical=None"

    def test_checkpointing_does_not_change_the_function(self) -> None:
        torch.manual_seed(0)
        model = FloodSegmenter(dataclasses.replace(_cfg("fusion_crossattn"), dropout=0.0))
        batch = _batch()
        plain, _ = model(**batch)
        plain.square().mean().backward()
        ref = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.enable_gradient_checkpointing(True)
        ckpt, _ = model(**batch)
        ckpt.square().mean().backward()
        assert torch.allclose(plain, ckpt, atol=1e-6)
        for name, g in ref.items():
            assert torch.allclose(g, dict(model.named_parameters())[name].grad, atol=1e-5), name

    def test_channels_last_inputs_are_accepted(self) -> None:
        model = FloodSegmenter(_cfg("fusion_crossattn")).to(memory_format=torch.channels_last).eval()
        batch = {k: (v.to(memory_format=torch.channels_last) if v is not None and v.dim() == 4 else v)
                 for k, v in _batch().items()}
        assert model(**batch)[0].shape == (B, 1, S, S)

    def test_head_starts_at_the_prior(self) -> None:
        model = FloodSegmenter(_cfg("fusion_crossattn")).eval()
        logits, _ = model(**_batch())
        assert torch.sigmoid(logits).mean().item() == pytest.approx(0.02, abs=0.02)
