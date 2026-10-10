"""Optimiser groups, schedule and per-sample modality dropout (docs/08 7.1, docs/07 7.3)."""

from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import ModelConfig, TrainConfig
from floodfusion.models.zoo import build_model
from floodfusion.optim import (
    apply_modality_dropout,
    build_optimizer,
    build_scheduler,
    learning_rate_factor,
    parameter_groups,
)
from floodfusion.train import model_inputs

SMALL = ModelConfig(encoder_channels=(16, 32, 64, 128), decoder_channels=(64, 32, 16, 8), attention_heads=2,
                    attention_head_dim=16, norm_groups=4)


class TestModalityDropout:
    def test_is_drawn_per_sample(self) -> None:
        """Per-batch dropping would make every row all-True or all-False."""
        g = torch.Generator().manual_seed(0)
        draws = torch.stack([apply_modality_dropout(torch.ones(16, dtype=torch.bool), 0.5, generator=g)
                             for _ in range(400)])
        per_batch = draws.float().mean(dim=1)
        assert ((per_batch > 0) & (per_batch < 1)).float().mean() > 0.95
        assert draws.float().mean().item() == pytest.approx(0.5, abs=0.03)

    def test_never_resurrects_missing_optical(self) -> None:
        available = torch.tensor([True, False, True, False])
        for seed in range(50):
            out = apply_modality_dropout(available, 0.3, generator=torch.Generator().manual_seed(seed))
            assert not bool((out & ~available).any())

    def test_p_zero_is_an_identity_copy(self) -> None:
        available = torch.tensor([True, False])
        out = apply_modality_dropout(available, 0.0)
        assert torch.equal(out, available) and out.data_ptr() != available.data_ptr()

    def test_input_is_not_mutated_and_validated(self) -> None:
        available = torch.ones(8, dtype=torch.bool)
        apply_modality_dropout(available, 1.0)
        assert bool(available.all())
        with pytest.raises(TypeError):
            apply_modality_dropout(torch.ones(4), 0.5)
        with pytest.raises(ValueError):
            apply_modality_dropout(available, 1.5)

    def test_force_sar_only_inputs(self) -> None:
        batch = {"sar": torch.zeros(2, 5, 8, 8), "terrain": torch.zeros(2, 3, 8, 8),
                 "optical": torch.zeros(2, 4, 8, 8), "optical_quality": torch.ones(2, 1, 8, 8)}
        out = model_inputs(batch, torch.ones(2, dtype=torch.bool), force_sar_only=True)
        assert out["optical"] is None and not bool(out["optical_available"].any())


class TestParameterGroups:
    def test_no_decay_on_norm_bias_and_embedding(self) -> None:
        model = build_model(SMALL)
        decay, no_decay = parameter_groups(model, 0.01)
        names = {id(p): n for n, p in model.named_parameters()}
        no_decay_names = {names[id(p)] for p in no_decay["params"]}
        decay_names = {names[id(p)] for p in decay["params"]}
        # Decaying the embedding pulls "optical absent" toward "optical present and zero" (docs/07 7.2).
        assert "missing_optical.embedding" in no_decay_names
        assert "fusion.gate.out.weight" in decay_names and "fusion.gate.out.bias" in no_decay_names
        assert all(not n.endswith(".bias") for n in decay_names)
        norm_params = {id(p) for m in model.modules() if isinstance(m, torch.nn.GroupNorm | torch.nn.LayerNorm)
                       for p in m.parameters()}
        assert norm_params <= {id(p) for p in no_decay["params"]}
        assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0

    def test_every_trainable_parameter_exactly_once(self) -> None:
        model = build_model(SMALL)
        groups = parameter_groups(model, 0.01)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == sum(1 for p in model.parameters() if p.requires_grad)

    def test_optimizer_uses_the_groups(self) -> None:
        opt = build_optimizer(build_model(SMALL), TrainConfig(epochs=3, warmup_epochs=1))
        assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]


class TestSchedule:
    def test_warmup_is_linear_and_cosine_ends_at_the_floor(self) -> None:
        f = [learning_rate_factor(s, warmup_steps=10, total_steps=100, min_ratio=0.01) for s in range(100)]
        assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
        assert all(a >= b for a, b in itertools.pairwise(f[10:]))
        assert learning_rate_factor(100, warmup_steps=10, total_steps=100, min_ratio=0.01) == pytest.approx(0.01)

    def test_cosine_midpoint(self) -> None:
        mid = learning_rate_factor(55, warmup_steps=10, total_steps=100, min_ratio=0.0)
        assert mid == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))

    def test_scheduler_steps_per_optimiser_step(self) -> None:
        model = build_model(SMALL)
        cfg = TrainConfig(epochs=4, warmup_epochs=1, learning_rate=1e-3, min_learning_rate=1e-5)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg, steps_per_epoch=5)
        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] == pytest.approx(2e-4) and lrs[4] == pytest.approx(1e-3)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-5, rel=1e-3)

    def test_rejects_an_empty_epoch(self) -> None:
        opt = build_optimizer(build_model(SMALL), TrainConfig(epochs=3, warmup_epochs=1))
        with pytest.raises(ValueError):
            build_scheduler(opt, TrainConfig(epochs=3, warmup_epochs=1), steps_per_epoch=0)
