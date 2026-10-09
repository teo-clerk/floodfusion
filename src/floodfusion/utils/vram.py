r"""Analytic VRAM budgeting and measurement.

``P1-M4.1`` requires a budget table **written and committed before the first long run**,
with the measured peak landing within 15 % of your prediction. Being wrong about the
budget is allowed; not having predicted is not.

The five consumers
------------------
========= ============================ ================================ ==========
\#        consumer                     scales with                      persistent
========= ============================ ================================ ==========
1         parameters                   :math:`P`                        yes
2         gradients                    :math:`P`                        yes
3         optimizer states             :math:`P`                        yes
4         **activations**              :math:`P \times B \times H \times W`  per step
5         workspace + fragmentation    implementation                   varies
========= ============================ ================================ ==========

**AdamW costs 16 bytes per parameter** (fp32 params + grads + two moments). At
:math:`P = 19` M that is 304 MB — small against a 7.6 GiB budget. This is the opposite
of the LLM regime, and it says immediately that **activations are the entire problem**,
so the levers that matter are batch size, chip size and checkpointing, not optimiser
choice.

Note that bf16 ``autocast`` keeps **fp32 master weights**: it casts per operation, it
does not convert your parameters. Budget them at 4 bytes, not 2.

References
----------
``docs/08_training_recipe_and_vram.md`` section 4.
``00_FOUNDATIONS/09_rtx4060_vram_playbook.md`` — the general method this specialises.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = ["MemoryBudget", "estimate_activation_bytes", "measure_peak", "static_bytes"]

#: Bytes per parameter for common optimisers, counting parameters, gradients and states.
OPTIMIZER_BYTES_PER_PARAM: dict[str, int] = {
    "adamw": 16,  # 4 param + 4 grad + 4 exp_avg + 4 exp_avg_sq
    "adam": 16,
    "sgd_momentum": 12,  # 4 + 4 + 4
    "sgd": 8,  # 4 + 4
}


@dataclass(frozen=True)
class MemoryBudget:
    """One row of the ``P1-M4`` budget table."""

    label: str
    n_params: int
    batch_size: int
    chip_size: int
    static_bytes: int
    activation_bytes_analytic: int
    overhead_factor: float
    measured_bytes: int | None = None

    @property
    def predicted_bytes(self) -> int:
        """Calibrated prediction: analytic activations times the overhead factor."""
        return int(
            self.static_bytes + self.activation_bytes_analytic * self.overhead_factor
        )

    @property
    def relative_error(self) -> float | None:
        """Signed relative error of the prediction. ``P1-M4`` requires ``|error| < 0.15``."""
        if self.measured_bytes is None:
            return None
        return (self.predicted_bytes - self.measured_bytes) / self.measured_bytes


def static_bytes(n_params: int, optimizer: str = "adamw") -> int:
    """Parameters + gradients + optimizer states, in bytes.

    TODO(student)
    -------------
    1. Look the optimiser up in :data:`OPTIMIZER_BYTES_PER_PARAM` and multiply.
    2. Raise a clear error for an unknown optimiser rather than defaulting silently — a
       silently wrong budget is worse than no budget.
    """
    raise NotImplementedError("Implement the static budget. See docs/08 section 4.2.")


def estimate_activation_bytes(
    stage_shapes: tuple[tuple[int, int, int, int], ...],
    *,
    tensors_per_stage: int = 4,
    bytes_per_element: int = 2,
) -> int:
    r"""Analytic activation estimate.

    Parameters
    ----------
    stage_shapes
        ``(B, C, H, W)`` for each stage that stores activations.
    tensors_per_stage
        Approximate number of stored tensors per conv--norm--activation block.
    bytes_per_element
        2 under bf16 ``autocast``.

    Notes
    -----
    **The counting rule**: count one stored tensor per operation with a non-trivial
    derivative with respect to its input — convolutions store their input,
    normalisations store input and an intermediate, non-linearities store input or
    output, and every concatenation result is stored. Roughly 4 per conv--norm--act
    block.

    .. warning::

       **This is a lower bound, typically by a factor of 1.5--2.5.** The gap comes from
       allocator reservation (``max_memory_reserved`` exceeds ``max_memory_allocated``),
       cuDNN convolution workspace, the autocast casting cache, under-counting by the
       "4 tensors" approximation, and evaluation-time spikes.

       The professional method — and what ``P1-M4`` is really teaching — is:

       1. compute :math:`M_{\mathrm{analytic}}`;
       2. run **one** short step and measure;
       3. calibrate :math:`k = M_{\mathrm{measured}} / M_{\mathrm{analytic}}`;
       4. use :math:`k` to predict every other configuration, and check.

       The 15 % criterion applies to your *calibrated* prediction, not to the raw
       analytic estimate. This is exactly how you would size a job on a cluster you are
       billed for.

    TODO(student)
    -------------
    1. Sum :math:`B\,C\,H\,W` over stages, times ``tensors_per_stage`` and
       ``bytes_per_element``.
    2. Reproduce the worked example of ``docs/08`` section 4.4 and check you get
       :math:`\approx 0.47` GiB for the reference model at :math:`B = 8`, :math:`256^2`.
    3. Then measure, calibrate :math:`k`, and complete the table of ``docs/08``
       section 4.6.
    """
    raise NotImplementedError("Implement the activation estimate. See docs/08 section 4.3.")


@contextmanager
def measure_peak(device: int = 0) -> Iterator[dict[str, int]]:
    """Context manager reporting peak allocated and reserved bytes.

    Yields
    ------
    dict[str, int]
        Populated on exit with ``allocated`` and ``reserved``.

    Notes
    -----
    Report **both**. A large gap between allocated and reserved indicates allocator
    fragmentation rather than genuine demand, which is a different problem with a
    different fix (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` rather than a
    smaller batch). Logging only ``allocated`` hides the distinction, and debugging
    challenge C2 in ``docs/10`` section 3 turns on exactly it.

    TODO(student)
    -------------
    1. ``torch.cuda.reset_peak_memory_stats(device)`` on entry.
    2. ``torch.cuda.synchronize(device)`` before reading — the peak is not final until
       queued kernels have run.
    3. Populate the yielded dict on exit.
    """
    raise NotImplementedError("Implement the peak probe. See docs/08 section 4.5.")
