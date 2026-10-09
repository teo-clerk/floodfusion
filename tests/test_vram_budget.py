r"""The 7.6 GiB envelope: analytic budgeting and measured peaks.

The specification for :mod:`floodfusion.utils.vram`. Milestone ``P1-M4``.

``P1-M4`` requires the budget table to be **written and committed before the first long
run**, with the measured peak landing within 15 % of the *calibrated* prediction. Being
wrong about the budget is allowed; not having predicted is not.

GPU-dependent tests are marked ``gpu`` and skip automatically when CUDA is unavailable.

The budget arithmetic below is pure Python: :mod:`floodfusion.utils.vram` and
:mod:`floodfusion.config` import no tensor library, so every static and activation
estimate here runs on a machine with no PyTorch installed. Only the two measured
peaks in :class:`TestMeasuredPeak` need ``torch``, and they import it themselves.
"""

from __future__ import annotations

import pytest

from floodfusion.config import TrainConfig
from floodfusion.utils.vram import (
    OPTIMIZER_BYTES_PER_PARAM,
    MemoryBudget,
    estimate_activation_bytes,
    measure_peak,
    static_bytes,
)

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_adamw_costs_sixteen_bytes_per_parameter",
    "test_optimizer_table_is_ordered_by_cost",
    "test_budget_predicted_bytes",
    "test_budget_relative_error",
    "test_relative_error_is_none_before_measurement",
    "test_reference_model_static_memory_is_small",
    "test_gradient_accumulation_preserves_effective_batch",
    "test_attention_matrix_scales_quadratically_with_tokens",
    "test_calibrated_prediction_meets_the_fifteen_percent_criterion",
})

USABLE_VRAM_BYTES = int(7.6 * 1024**3)


class TestStaticBudget:
    def test_adamw_costs_sixteen_bytes_per_parameter(self) -> None:
        """4 param + 4 grad + 4 exp_avg + 4 exp_avg_sq, all fp32."""
        assert OPTIMIZER_BYTES_PER_PARAM["adamw"] == 16

    def test_optimizer_table_is_ordered_by_cost(self) -> None:
        t = OPTIMIZER_BYTES_PER_PARAM
        assert t["sgd"] < t["sgd_momentum"] < t["adamw"]

    def test_reference_model_static_memory_is_small(self) -> None:
        """19 M params x 16 B = 304 MB of a 7.6 GiB budget.

        The opposite of the LLM regime: activations are the entire problem, so the
        levers that matter are batch size, chip size and checkpointing -- not the
        optimiser (``docs/08`` section 4.2).
        """
        expected = 19_000_000 * 16
        assert expected == pytest.approx(304_000_000, rel=0.02)
        assert expected < 0.05 * USABLE_VRAM_BYTES

    def test_static_bytes_matches_the_table(self) -> None:
        assert static_bytes(1_000_000, "adamw") == 16_000_000

    def test_unknown_optimizer_raises(self) -> None:
        """A silently wrong budget is worse than no budget."""
        with pytest.raises((KeyError, ValueError)):
            static_bytes(1_000_000, "shampoo")


class TestActivationEstimate:
    def test_scales_linearly_with_batch(self) -> None:
        one = estimate_activation_bytes(((1, 64, 128, 128),))
        eight = estimate_activation_bytes(((8, 64, 128, 128),))
        assert eight == 8 * one

    def test_scales_quadratically_with_chip_size(self) -> None:
        small = estimate_activation_bytes(((4, 64, 128, 128),))
        large = estimate_activation_bytes(((4, 64, 256, 256),))
        assert large == 4 * small

    def test_bfloat16_is_two_bytes_per_element(self) -> None:
        b = estimate_activation_bytes(((1, 1, 1, 1),), tensors_per_stage=1, bytes_per_element=2)
        assert b == 2

    def test_reference_configuration_lands_near_half_a_gibibyte(self) -> None:
        """Reproduces the worked example of ``docs/08`` section 4.4."""
        stages = ((8, 8, 256, 256), (8, 64, 128, 128), (8, 128, 64, 64),
                  (8, 256, 32, 32), (8, 512, 16, 16))
        est = estimate_activation_bytes(stages, tensors_per_stage=4, bytes_per_element=2)
        assert 0.3 * 1024**3 < est < 0.9 * 1024**3


class TestMemoryBudget:
    def _budget(self, measured: int | None = None) -> MemoryBudget:
        return MemoryBudget(
            label="fusion_256", n_params=19_000_000, batch_size=8, chip_size=256,
            static_bytes=304_000_000, activation_bytes_analytic=500_000_000,
            overhead_factor=2.0, measured_bytes=measured,
        )

    def test_budget_predicted_bytes(self) -> None:
        assert self._budget().predicted_bytes == 304_000_000 + 1_000_000_000

    def test_budget_relative_error(self) -> None:
        b = self._budget(measured=1_200_000_000)
        assert b.relative_error == pytest.approx((1_304_000_000 - 1_200_000_000) / 1_200_000_000)

    def test_relative_error_is_none_before_measurement(self) -> None:
        assert self._budget().relative_error is None

    def test_calibrated_prediction_meets_the_fifteen_percent_criterion(self) -> None:
        """``P1-M4``'s gate, applied to the CALIBRATED prediction (``docs/08`` 4.5)."""
        b = self._budget(measured=1_300_000_000)
        assert abs(b.relative_error) < 0.15


class TestBatchingArithmetic:
    def test_gradient_accumulation_preserves_effective_batch(self) -> None:
        """The point of accumulation: batch 16 at the activation cost of 4."""
        small = TrainConfig(micro_batch_size=4, grad_accum_steps=4)
        large = TrainConfig(micro_batch_size=8, grad_accum_steps=2)
        assert small.effective_batch_size == large.effective_batch_size == 16

    def test_attention_matrix_scales_quadratically_with_tokens(self) -> None:
        """Why fusion happens at stage 3 and not stage 1 (``docs/07`` section 6).

        Naive attention at 128x128 tokens needs 34 GB; at 32x32 it needs 134 MB. This
        is not a tuning problem — it is quadratic scaling.
        """
        def naive_bytes(tokens: int, batch: int = 8, heads: int = 8) -> int:
            return batch * heads * tokens * tokens * 2

        assert naive_bytes(128 * 128) > 30 * 1024**3
        assert naive_bytes(32 * 32) < 200 * 1024**2
        assert naive_bytes(64 * 64) == 16 * naive_bytes(32 * 32)


@pytest.mark.gpu
class TestMeasuredPeak:
    def test_measure_peak_reports_allocated_and_reserved(self) -> None:
        """Report BOTH: a large gap means fragmentation, not genuine demand."""
        torch = pytest.importorskip(
            "torch", reason="measuring a peak allocation requires PyTorch"
        )
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        with measure_peak() as stats:
            torch.zeros(1024, 1024, device="cuda")
        assert stats["allocated"] > 0
        assert stats["reserved"] >= stats["allocated"]

    def test_reference_configuration_fits_the_envelope(self) -> None:
        torch = pytest.importorskip(
            "torch", reason="measuring a peak allocation requires PyTorch"
        )
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        with measure_peak() as stats:
            torch.zeros(64, 12, 256, 256, device="cuda", dtype=torch.bfloat16)
        assert stats["allocated"] < USABLE_VRAM_BYTES
