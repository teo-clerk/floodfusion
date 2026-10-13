r"""Low-rank adaptation (LoRA, Hu et al., 2022) of a frozen network's linear layers — for the fair TerraMind (v1.2).

A wrapped layer computes :math:`W x + b + \tfrac{\alpha}{r} B A x`. Here :math:`W, b` are frozen, :math:`A`
(r × in) is random, and :math:`B` (out × r) starts at **zero**, so the adapted network is the frozen network,
exactly, until training moves :math:`B`. Only :math:`A` and :math:`B` train: a few hundred thousand parameters
instead of a whole ViT. That is what lets TerraMind-base adapt on an 8 GB GPU.

In-house rather than ``peft``: ``peft`` would pull ``transformers`` and ``accelerate`` into the lock for about forty
lines of arithmetic, and its wrapping rules are harder to pin in a test.

TerraMind's encoder blocks name their layers ``attn.qkv`` (one fused projection), ``attn.proj`` and
``mlp.fc1`` / ``mlp.fc2`` / ``mlp.fc3`` (gated MLP); :data:`DEFAULT_TARGETS` adapts the attention projections, the
usual choice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

__all__ = ["DEFAULT_TARGETS", "LoRALinear", "apply_lora", "lora_parameters"]

DEFAULT_TARGETS = ("attn.qkv", "attn.proj")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update, zero at initialisation."""

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype)
        return self.base(x) + self.scale * update


def apply_lora(model: nn.Module, *, targets: Iterable[str] = DEFAULT_TARGETS, rank: int = 8, alpha: float = 16.0,
               dropout: float = 0.0) -> list[str]:
    """Replace every ``nn.Linear`` whose qualified name ends with a target suffix; return the names replaced."""
    suffixes = tuple(targets)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.endswith(suffixes):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full)
    if not replaced:
        raise ValueError(f"no nn.Linear ends with any of {suffixes}")
    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The trainable low-rank factors of every :class:`LoRALinear` in ``model``."""
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.lora_a, m.lora_b)]
