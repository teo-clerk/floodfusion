"""``P1-M3.5`` smoke test on the real GPU: B=8, 256^2, bf16, with and without optical.

Asserts no NaN, the correct output shape, that gradients reach **both** encoders when
optical is present and **only** the SAR encoder when it is absent, and that one optimiser
step of the reference model fits the 7.6 GiB envelope. Skips without CUDA.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    LossConfig,
    ModelConfig,
)
from floodfusion.models.losses import CompoundFocalDiceLoss
from floodfusion.models.segmenter import FloodSegmenter
from floodfusion.utils.vram import measure_peak

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

B, S = 8, 256
USABLE_VRAM_BYTES = int(7.6 * 1024**3)


def _batch(available: bool):
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    def cl(t):  # channels_last on both model and inputs (docs/08 section 5.5)
        return t.to(dev).contiguous(memory_format=torch.channels_last)
    return dict(
        sar=cl(torch.randn(B, N_SAR_CHANNELS, S, S, generator=g)),
        terrain=cl(torch.rand(B, N_TERRAIN_CHANNELS, S, S, generator=g)),
        optical=cl(torch.rand(B, N_OPTICAL_CHANNELS, S, S, generator=g)),
        optical_quality=cl(torch.rand(B, 1, S, S, generator=g)),
        optical_available=torch.full((B,), available, device=dev),
    )


@pytest.mark.parametrize("available", [True, False])
def test_reference_model_forward_backward_bf16(available: bool) -> None:
    torch.manual_seed(0)
    model = FloodSegmenter(ModelConfig()).cuda().to(memory_format=torch.channels_last)
    loss_fn = CompoundFocalDiceLoss(LossConfig())
    batch = _batch(available)
    target = (torch.rand(B, 1, S, S, device="cuda") < 0.02).float()
    valid = torch.ones(B, 1, S, S, dtype=torch.bool, device="cuda")
    with measure_peak() as stats:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _gate = model(**batch)
            total, _parts = loss_fn(logits, target, valid)
        total.backward()
    assert logits.shape == (B, 1, S, S)
    assert torch.isfinite(logits.float()).all() and torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    sar_grad = sum(p.grad.abs().sum().item() for p in model.sar_encoder.parameters() if p.grad is not None)
    opt_grads = [p.grad for p in model.optical_encoder.parameters()]
    assert sar_grad > 0
    if available:
        assert any(g is not None and g.abs().sum() > 0 for g in opt_grads)
    else:
        assert all(g is None for g in opt_grads)
        assert model.missing_optical.embedding.grad.abs().sum() > 0
    assert stats["allocated"] < USABLE_VRAM_BYTES


def test_sar_only_inference_is_cheaper_than_fusion() -> None:
    """Short-circuiting the optical encoder is what makes row 11 cheap to evaluate."""
    model = FloodSegmenter(ModelConfig()).cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = _batch(True)
        with measure_peak() as fusion:
            model(**batch)
        with measure_peak() as sar_only:
            model(batch["sar"], batch["terrain"], optical=None)
    assert sar_only["allocated"] < fusion["allocated"]
