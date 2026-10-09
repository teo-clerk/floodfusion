r"""Optimiser, schedule and modality dropout (``docs/08`` section 7.1, ``docs/07`` section 7.3).

- :func:`build_optimizer` — AdamW with decoupled weight decay applied to weight matrices and
  convolution kernels only: **never** to normalisation parameters, biases, the learned
  missing-modality embedding or any other 1-D parameter. Decaying a GroupNorm scale pulls it
  toward zero and silently shrinks every activation; decaying the missing-modality embedding
  pulls "optical absent" toward "optical present and all-zero", which is exactly the
  ambiguity the embedding exists to remove (``docs/07`` section 7.2).
- :func:`build_scheduler` — linear warmup then cosine annealing to ``min_learning_rate``,
  stepped **once per optimiser step** (not per micro-batch, not per epoch). Warmup matters:
  attention logits are unstable early (``docs/07`` section 4.2).
- :func:`apply_modality_dropout` — per sample, never on SAR, never turning a genuinely
  missing optical observation into an available one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from floodfusion.config import TrainConfig

__all__ = [
    "apply_modality_dropout",
    "build_optimizer",
    "build_scheduler",
    "learning_rate_factor",
    "parameter_groups",
]

#: Parameters whose name contains one of these are never decayed, whatever their shape.
_NO_DECAY_NAME_PARTS = ("missing_optical", "embedding", "blend_scale", "blend_bias")
_NORM_TYPES = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Two AdamW groups: decayed kernels and undecayed everything else.

    A parameter is decayed only if it is at least 2-D, belongs to neither a normalisation
    layer nor a learned embedding, and is not a bias.
    """
    norm_params = {id(p) for m in model.modules() if isinstance(m, _NORM_TYPES) for p in m.parameters(recurse=False)}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        exempt = (
            param.ndim < 2
            or name.endswith(".bias")
            or id(param) in norm_params
            or any(part in name for part in _NO_DECAY_NAME_PARTS)
        )
        (no_decay if exempt else decay).append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW over :func:`parameter_groups`; the fused CUDA kernel when every param is on CUDA."""
    groups = parameter_groups(model, train_cfg.weight_decay)
    params: Iterable[nn.Parameter] = (p for g in groups for p in g["params"])  # type: ignore[union-attr]
    on_cuda = all(p.is_cuda for p in params)
    return torch.optim.AdamW(groups, lr=train_cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                             fused=on_cuda and torch.cuda.is_available())


def learning_rate_factor(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    r"""Multiplier on the base LR at optimiser step ``step`` (0-indexed).

    Linear warmup from :math:`1/w` to 1 over ``warmup_steps``, then cosine from 1 down to
    ``min_ratio`` at ``total_steps``:
    :math:`f = r + (1-r)\,\tfrac12\bigl(1 + \cos(\pi\,\tfrac{s - w}{T - w})\bigr)`.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / span, 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine ``LambdaLR``; call ``.step()`` after every ``optimizer.step()``."""
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1 (fewer chips than one effective batch?)")
    total = train_cfg.epochs * steps_per_epoch
    warmup = train_cfg.warmup_epochs * steps_per_epoch
    min_ratio = train_cfg.min_learning_rate / train_cfg.learning_rate

    def factor(step: int) -> float:
        return learning_rate_factor(step, warmup_steps=warmup, total_steps=total, min_ratio=min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def apply_modality_dropout(
    optical_available: torch.Tensor,
    p_drop: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop the optical branch **per sample** with probability ``p_drop``.

    Returns a new ``(B,)`` bool tensor; the input is not modified. A sample is available
    after dropout iff it was available before **and** its own Bernoulli draw kept it, so a
    genuinely missing observation is never resurrected (``docs/07`` section 7.3). SAR has
    no counterpart here: it is never dropped.
    """
    if optical_available.dtype != torch.bool or optical_available.dim() != 1:
        raise TypeError("optical_available must be a (B,) bool tensor")
    if not 0.0 <= p_drop <= 1.0:
        raise ValueError("p_drop must lie in [0, 1]")
    if p_drop == 0.0:
        return optical_available.clone()
    draws = torch.rand(optical_available.shape, generator=generator, device="cpu")
    keep = (draws >= p_drop).to(optical_available.device)
    return optical_available & keep
