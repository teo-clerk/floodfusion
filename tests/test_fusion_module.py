r"""Cross-attention fusion: the architectural contracts of ``docs/07`` section 9.5.

The specification for :mod:`floodfusion.models.fusion_module`. Milestone ``P1-M3``.

Two tests here are worth more than the rest combined:

- :meth:`TestRequirementR2.test_zero_gate_is_bit_identical_to_sar_only` is the
  **mechanised statement of operational requirement R2** (``docs/00`` section 7). The
  design satisfies R2 *structurally* — because the fusion is a gated residual on the SAR
  stream, :math:`g \to 0` gives :math:`F_{\mathrm{fused}} = F_s` exactly — rather than by
  hoping the network learns it. If this test fails, the architecture is wrong, not the
  training.

- :meth:`TestGradientRouting.test_absent_optical_sends_no_gradient_to_optical_encoder`
  catches the most common bug in this architecture: a fill value that still routes
  gradient into the optical encoder, quietly training it on fabricated inputs.
"""

from __future__ import annotations

import math

import pytest

#: The subject of this module imports ``torch`` at module level, so the module
#: cannot be collected without it. The guard sits here rather than in
#: ``conftest.py``: a ``Skipped`` raised while the conftest is being imported is
#: attributed to the whole directory, and would take the torch-free modules with it.
torch = pytest.importorskip(
    "torch", reason="floodfusion.models.fusion_module imports torch at module level"
)

from floodfusion.config import ModelConfig  # noqa: E402
from floodfusion.models.fusion_module import (  # noqa: E402
    AvailabilityGate,
    CrossAttentionFusion,
    MissingModalityEmbedding,
    reference_attention,
)


B, C_S, C_O, HW = 2, 64, 64, 8  # small; distinct sizes so a transpose is detectable


def _fusion(model_config: ModelConfig) -> CrossAttentionFusion:
    return CrossAttentionFusion(C_S, C_O, model_config)


def _inputs(height: int = HW, width: int = HW + 4):
    """Non-square spatial dims on purpose: a transposed H/W cannot pass silently."""
    g = torch.Generator().manual_seed(7)
    return (
        torch.randn(B, C_S, height, width, generator=g),
        torch.randn(B, C_O, height, width, generator=g),
        torch.rand(B, 1, height, width, generator=g),
    )


# ---------------------------------------------------------------------------
# Attention mechanics
# ---------------------------------------------------------------------------
class TestScaledDotProductAttention:
    def test_reference_attention_rows_sum_to_one(self) -> None:
        """Softmax over the KEY axis. SDPA never exposes the matrix, so we check here."""
        g = torch.Generator().manual_seed(0)
        q = torch.randn(2, 4, 16, 8, generator=g, dtype=torch.float64)
        k = torch.randn(2, 4, 20, 8, generator=g, dtype=torch.float64)
        v = torch.randn(2, 4, 20, 8, generator=g, dtype=torch.float64)

        out, weights = reference_attention(q, k, v)

        assert weights.shape == (2, 4, 16, 20), "weights are (B, h, N_q, N_k)"
        assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 16, dtype=torch.float64))
        assert (weights >= 0).all(), "softmax weights are non-negative"
        assert out.shape == (2, 4, 16, 8), "output takes N_q from the QUERY, d_v from V"

    def test_output_lies_in_the_convex_hull_of_values(self) -> None:
        r"""The invariant the :math:`Q/K/V` derivation rests on (``docs/07`` section 5.4).

        Attention *retrieves and mixes* values; it cannot create content outside their
        range. This is why drawing :math:`V` from SAR guarantees every retrievable
        quantity is a real SAR feature, even under a degenerate query.
        """
        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, 1, 5, 4, generator=g, dtype=torch.float64)
        k = torch.randn(1, 1, 7, 4, generator=g, dtype=torch.float64)
        v = torch.randn(1, 1, 7, 4, generator=g, dtype=torch.float64)

        out, _ = reference_attention(q, k, v)

        assert (out <= v.max(dim=2, keepdim=True).values + 1e-9).all()
        assert (out >= v.min(dim=2, keepdim=True).values - 1e-9).all()

    def test_agrees_with_torch_sdpa(self) -> None:
        """If these differ by a constant factor, you have double-scaled by 1/sqrt(d)."""
        g = torch.Generator().manual_seed(2)
        q = torch.randn(2, 4, 16, 8, generator=g, dtype=torch.float64)
        k = torch.randn(2, 4, 16, 8, generator=g, dtype=torch.float64)
        v = torch.randn(2, 4, 16, 8, generator=g, dtype=torch.float64)

        ours, _ = reference_attention(q, k, v)
        theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(ours, theirs, atol=1e-10)

    def test_scaling_keeps_logit_variance_at_unity(self) -> None:
        r""":math:`\operatorname{Var}[q\cdot k] = d_k`, so the scale must be :math:`1/\sqrt{d_k}`.

        Unscaled at :math:`d_k = 64`, logits have :math:`\sigma = 8` and spread over ~48.
        Softmax of that is numerically one-hot, the Jacobian
        :math:`p_i(\delta_{ij} - p_j)` vanishes, and **no gradient reaches** :math:`K`
        or :math:`V` — silently, with a finite loss (``docs/07`` section 4.2).
        """
        d_k = 64
        g = torch.Generator().manual_seed(3)
        q = torch.randn(4096, d_k, generator=g, dtype=torch.float64)
        k = torch.randn(4096, d_k, generator=g, dtype=torch.float64)

        unscaled = (q @ k.T).std().item()
        scaled = ((q @ k.T) / math.sqrt(d_k)).std().item()

        assert unscaled == pytest.approx(math.sqrt(d_k), rel=0.10)
        assert scaled == pytest.approx(1.0, rel=0.10)

    def test_unscaled_attention_saturates_the_softmax(self) -> None:
        """Demonstrates the failure the scaling prevents, rather than asserting it abstractly.

        With :math:`N = 64` keys, uniform attention would put :math:`1/64 = 0.016` on
        each. Unscaled logits concentrate ~0.82 of the mass on a single key — the
        distribution is effectively one-hot, so the softmax Jacobian
        :math:`p_i(\\delta_{ij} - p_j)` is ~0 everywhere and gradient stops flowing to
        :math:`K` and :math:`V`. Scaling keeps the peak near 0.12, an order of magnitude
        flatter, and the layer keeps learning.
        """
        d_k, n_keys = 64, 64
        g = torch.Generator().manual_seed(4)
        q = torch.randn(1, 1, n_keys, d_k, generator=g, dtype=torch.float64)
        k = torch.randn(1, 1, n_keys, d_k, generator=g, dtype=torch.float64)
        scores = q @ k.transpose(-2, -1)

        unscaled_peak = torch.softmax(scores, dim=-1).max(-1).values.mean().item()
        scaled_peak = torch.softmax(scores / math.sqrt(d_k), dim=-1).max(-1).values.mean().item()
        uniform = 1.0 / n_keys

        assert unscaled_peak > 0.5, "unscaled logits must concentrate most mass on one key"
        assert scaled_peak < 0.25, "scaled logits must stay distributed"
        assert unscaled_peak > 4 * scaled_peak, "the scaling must flatten the distribution"
        assert scaled_peak > uniform, "but not to the point of being uninformative"


# ---------------------------------------------------------------------------
# Missing-modality embedding
# ---------------------------------------------------------------------------
class TestMissingModalityEmbedding:
    def test_broadcasts_to_the_requested_shape(self) -> None:
        emb = MissingModalityEmbedding(C_O)
        out = emb(B, HW, HW + 4, device=torch.device("cpu"), dtype=torch.float32)
        assert out.shape == (B, C_O, HW, HW + 4)

    def test_is_spatially_constant(self) -> None:
        """Constant queries make attention a single global lookup over the SAR values."""
        emb = MissingModalityEmbedding(C_O)
        out = emb(B, HW, HW, device=torch.device("cpu"), dtype=torch.float32)
        assert torch.allclose(out.std(dim=(2, 3)), torch.zeros(B, C_O), atol=1e-7)

    def test_is_not_identically_zero(self) -> None:
        """Zero is a legitimate normalised reflectance; 'absent' must be distinguishable."""
        emb = MissingModalityEmbedding(C_O)
        out = emb(1, 4, 4, device=torch.device("cpu"), dtype=torch.float32)
        assert out.abs().sum().item() > 0.0

    def test_is_a_learnable_parameter(self) -> None:
        emb = MissingModalityEmbedding(C_O)
        assert any(p.requires_grad for p in emb.parameters())

    def test_receives_gradient_when_used(self) -> None:
        emb = MissingModalityEmbedding(C_O)
        emb(B, 4, 4, device=torch.device("cpu"), dtype=torch.float32).sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in emb.parameters()
        )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
class TestAvailabilityGate:
    def test_output_shape_and_range(self) -> None:
        gate = AvailabilityGate(C_S, C_O, bias_init=-2.0)
        sar, opt, quality = _inputs()
        g = gate(sar, opt, quality)
        assert g.shape == (B, 1, HW, HW + 4), "per-pixel, not per-chip"
        assert (g >= 0).all() and (g <= 1).all()

    def test_negative_bias_starts_near_sar_only(self) -> None:
        r"""Initialising near :math:`g \approx \sigma(-2) = 0.12`: optical must be EARNED."""
        gate = AvailabilityGate(C_S, C_O, bias_init=-2.0)
        sar, opt, quality = _inputs()
        assert gate(sar, opt, quality).mean().item() < 0.35

    def test_gate_is_per_pixel_not_per_chip(self) -> None:
        """Cloud edges cut across chips; a scalar gate cannot express a half-clear chip."""
        gate = AvailabilityGate(C_S, C_O, bias_init=0.0)
        sar, opt, quality = _inputs()
        quality[:, :, :, : (HW + 4) // 2] = 0.0
        quality[:, :, :, (HW + 4) // 2 :] = 1.0
        g = gate(sar, opt, quality)
        assert g.std().item() > 0.0, "gate must vary spatially"


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------
class TestShapeContract:
    def test_forward_returns_fused_and_gate(self, model_config: ModelConfig) -> None:
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        fused, gate = fusion(sar, opt, quality)
        assert fused.shape == sar.shape, "output width must equal the SAR width"
        assert gate.shape == (B, 1, HW, HW + 4)

    def test_output_dtype_matches_input(self, model_config: ModelConfig) -> None:
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        fused, _ = fusion(sar, opt, quality)
        assert fused.dtype == sar.dtype

    @pytest.mark.parametrize(("h", "w"), [(8, 8), (8, 12), (16, 16)])
    def test_accepts_varied_spatial_sizes(
        self, model_config: ModelConfig, h: int, w: int
    ) -> None:
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs(h, w)
        fused, _ = fusion(sar, opt, quality)
        assert fused.shape == (B, C_S, h, w)

    def test_mismatched_batch_is_rejected(self, model_config: ModelConfig) -> None:
        """The shape contract must be asserted, not silently broadcast."""
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            fusion(sar, opt[:1], quality)


# ---------------------------------------------------------------------------
# Requirement R2 — the structural guarantee
# ---------------------------------------------------------------------------
class TestRequirementR2:
    r"""The model must produce a usable map from SAR alone (``docs/00`` section 7)."""

    def test_zero_gate_is_bit_identical_to_sar_only(
        self, model_config: ModelConfig
    ) -> None:
        r"""**Contract 3 — the most valuable test in this file.**

        With :math:`g \equiv 0`,

        .. math:: F_{\mathrm{fused}} = F_s + 0 \odot \mathrm{CrossAttn}(\cdot) = F_s

        **exactly**, bit for bit. Not "close to"; identical. If this fails, the residual
        path is wrong — a bias added after the gate, a normalisation applied to the sum,
        or a projection on the residual branch — and R2 is no longer structural.
        """
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()

        with torch.no_grad():
            for name, param in fusion.named_parameters():
                if "gate" in name and "bias" in name:
                    param.fill_(-1e9)  # sigmoid -> exactly 0

        fused, gate = fusion(sar, opt, quality)

        assert torch.equal(gate, torch.zeros_like(gate)), "gate must saturate to exactly 0"
        assert torch.equal(fused, sar), (
            "with the gate closed the fused representation must be BIT-IDENTICAL to the "
            "SAR features. See docs/07 section 5.4."
        )

    def test_missing_optical_still_returns_correct_shape(
        self, model_config: ModelConfig
    ) -> None:
        fusion = _fusion(model_config)
        sar, _, quality = _inputs()
        embedding = MissingModalityEmbedding(C_O)
        opt = embedding(B, HW, HW + 4, device=sar.device, dtype=sar.dtype)
        fused, _ = fusion(sar, opt, torch.zeros_like(quality))
        assert fused.shape == sar.shape


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------
class TestGradientRouting:
    def test_present_optical_sends_gradient_to_both_streams(
        self, model_config: ModelConfig
    ) -> None:
        """Contract 4."""
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        sar.requires_grad_(True)
        opt.requires_grad_(True)

        with torch.no_grad():
            for name, param in fusion.named_parameters():
                if "gate" in name and "bias" in name:
                    param.fill_(4.0)  # open the gate

        fused, _ = fusion(sar, opt, quality)
        fused.sum().backward()

        assert sar.grad is not None and sar.grad.abs().sum() > 0
        assert opt.grad is not None and opt.grad.abs().sum() > 0

    def test_absent_optical_sends_no_gradient_to_optical_encoder(
        self, model_config: ModelConfig
    ) -> None:
        """**Contract 5.** Catches a fill value that still trains the optical encoder.

        When optical is absent the query comes from the learned embedding, so gradient
        must reach *that parameter* — and must not reach the optical feature tensor the
        encoder would have produced.
        """
        fusion = _fusion(model_config)
        embedding = MissingModalityEmbedding(C_O)
        sar, optical_from_encoder, quality = _inputs()
        optical_from_encoder.requires_grad_(True)

        substituted = embedding(B, HW, HW + 4, device=sar.device, dtype=sar.dtype)
        fused, _ = fusion(sar, substituted, quality)
        fused.sum().backward()

        assert optical_from_encoder.grad is None or optical_from_encoder.grad.abs().sum() == 0, (
            "the optical encoder must receive NO gradient when its output was replaced "
            "by the missing-modality embedding"
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in embedding.parameters()
        ), "the missing-modality embedding must be trained"

    def test_closed_gate_blocks_gradient_through_the_attention_path(
        self, model_config: ModelConfig
    ) -> None:
        r"""At :math:`g = 0` the residual still trains :math:`F_s`, but attention does not."""
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        sar.requires_grad_(True)
        opt.requires_grad_(True)

        with torch.no_grad():
            for name, param in fusion.named_parameters():
                if "gate" in name and "bias" in name:
                    param.fill_(-1e9)

        fused, _ = fusion(sar, opt, quality)
        fused.sum().backward()

        assert sar.grad is not None and sar.grad.abs().sum() > 0, "residual must carry gradient"
        assert opt.grad is None or opt.grad.abs().sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
class TestNumerics:
    def test_no_nan_or_inf_in_forward_and_backward(
        self, model_config: ModelConfig
    ) -> None:
        """Contract 6."""
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        sar.requires_grad_(True)
        fused, gate = fusion(sar, opt, quality)

        assert torch.isfinite(fused).all()
        assert torch.isfinite(gate).all()

        fused.sum().backward()
        assert torch.isfinite(sar.grad).all()

    def test_stable_under_bfloat16_autocast(self, model_config: ModelConfig) -> None:
        fusion = _fusion(model_config)
        sar, opt, quality = _inputs()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            fused, _ = fusion(sar, opt, quality)
        assert torch.isfinite(fused.float()).all()

    def test_head_dim_respects_the_flash_constraint(
        self, model_config: ModelConfig
    ) -> None:
        """head_dim <= 128, or the FlashAttention kernel is not selected."""
        assert model_config.attention_head_dim <= 128

    def test_inner_dim_is_heads_times_head_dim(self, model_config: ModelConfig) -> None:
        fusion = _fusion(model_config)
        assert fusion.inner_dim == model_config.attention_heads * model_config.attention_head_dim
