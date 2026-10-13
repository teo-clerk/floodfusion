"""In-house LoRA for the fair TerraMind (v1.2): identity at initialisation, only the low-rank factors train."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from floodfusion.models.lora import LoRALinear, apply_lora, lora_parameters


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(16, 48), nn.Linear(16, 16)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(16, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn.proj(self.attn.qkv(x)[..., :16]) + self.mlp.fc1(x)[..., :16]


def _encoder() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(_Block(), _Block())


def test_adapted_network_equals_the_frozen_one_at_initialisation() -> None:
    model, x = _encoder(), torch.randn(4, 10, 16)
    before = model(x)
    replaced = apply_lora(model, rank=4)
    assert replaced == ["0.attn.qkv", "0.attn.proj", "1.attn.qkv", "1.attn.proj"]   # mlp.fc1 left alone
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)


def test_only_the_low_rank_factors_train_and_they_move_the_output() -> None:
    model, x = _encoder(), torch.randn(4, 10, 16)
    apply_lora(model, rank=4)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_a", "lora_b")) or ".mlp." in n for n in trainable)
    params = lora_parameters(model)
    assert len(params) == 8 and sum(p.numel() for p in params) == 2 * (4 * 16 + 48 * 4) + 2 * (4 * 16 + 16 * 4)
    before = model(x).detach()
    opt = torch.optim.SGD(params, lr=0.1)
    model(x).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(model(x), before)
    assert all(p.grad is None for n, p in model.named_parameters() if ".base." in n)


def test_bf16_autocast_and_bad_arguments() -> None:
    layer = LoRALinear(nn.Linear(8, 8), rank=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert layer(torch.randn(3, 8)).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(8, 8), rank=0)
    with pytest.raises(ValueError, match=r"no nn\.Linear"):
        apply_lora(nn.Sequential(nn.Conv2d(1, 1, 1)))
