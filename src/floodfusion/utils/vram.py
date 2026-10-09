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

Implementation notes
--------------------
This module imports **no tensor library at module level**: the arithmetic half of
``tests/test_vram_budget.py`` must run on a machine without PyTorch. ``torch`` is imported
inside :func:`measure_peak`, :func:`collect_block_shapes` and the calibration CLI.

The calibration CLI (``python -m floodfusion.utils.vram``) implements the professional
method of ``docs/08`` section 4.5: compute the analytic estimate, measure **one** step of
the reference configuration, calibrate :math:`k = M_{\mathrm{measured}} /
M_{\mathrm{analytic}}`, then predict every other configuration of the section 4.6 table
*before* measuring it, and report the error.

References
----------
``docs/08_training_recipe_and_vram.md`` section 4.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch

__all__ = [
    "OPTIMIZER_BYTES_PER_PARAM",
    "MemoryBudget",
    "collect_block_shapes",
    "estimate_activation_bytes",
    "load_overhead_factor",
    "measure_peak",
    "predict_training_peak",
    "static_bytes",
]

LOGGER = logging.getLogger(__name__)

#: Bytes per parameter for common optimisers, counting parameters, gradients and states.
OPTIMIZER_BYTES_PER_PARAM: dict[str, int] = {
    "adamw": 16,  # 4 param + 4 grad + 4 exp_avg + 4 exp_avg_sq
    "adam": 16,
    "sgd_momentum": 12,  # 4 + 4 + 4
    "sgd": 8,  # 4 + 4
}

#: Midpoint of the 1.5--2.5 lower-bound gap of docs/08 section 4.5, used only until a
#: measured calibration factor exists.
DEFAULT_OVERHEAD_FACTOR: float = 2.0


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
    """Parameters + gradients + optimizer states, in bytes (``docs/08`` section 4.2).

    Raises
    ------
    ValueError
        For an optimiser missing from :data:`OPTIMIZER_BYTES_PER_PARAM`. A silently wrong
        budget is worse than no budget, so there is no default.
    """
    key = optimizer.lower()
    if key not in OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown optimiser {optimizer!r}; known: {sorted(OPTIMIZER_BYTES_PER_PARAM)}"
        )
    if n_params < 0:
        raise ValueError("n_params must be non-negative")
    return int(n_params) * OPTIMIZER_BYTES_PER_PARAM[key]


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def estimate_activation_bytes(
    stage_shapes: tuple[tuple[int, int, int, int], ...],
    *,
    tensors_per_stage: int = 4,
    bytes_per_element: int = 2,
    decoder_mirror: bool = True,
) -> int:
    r"""Analytic activation estimate for a U-Net (``docs/08`` sections 4.3--4.4).

    Parameters
    ----------
    stage_shapes
        With ``decoder_mirror=True``: the **encoder pyramid** ``(B, C, H, W)``, finest
        first, **bottleneck last** — e.g. the input plus :math:`s_1 \ldots s_4`. With
        ``decoder_mirror=False``: every block that stores activations, enumerated
        explicitly (e.g. by :func:`collect_block_shapes`).
    tensors_per_stage
        Stored tensors per conv--norm--act block (the "4 per block" rule).
    bytes_per_element
        2 under bf16 ``autocast``.
    decoder_mirror
        Count the U-Net decoder implied by the pyramid: every level except the bottleneck
        is revisited by a decoder block whose input is the concatenation of the upsampled
        path and the skip, i.e. roughly **twice** the encoder width at that level
        (``docs/08`` section 4.4: "skip concatenations roughly doubling channel counts").

    Returns
    -------
    int
        Bytes, computed in exact integer arithmetic, so the estimate scales *exactly*
        linearly with batch and quadratically with chip size.

    Notes
    -----
    **The counting rule**: one stored tensor per operation with a non-trivial derivative
    with respect to its input — convolutions store their input, normalisations their
    input and an intermediate, non-linearities their input or output, and every
    concatenation result is stored. Roughly 4 per conv--norm--act block.

    With the mirror term the reference pyramid of ``docs/08`` section 4.4 at
    :math:`B = 8`, :math:`256^2` gives 0.43 GiB against the worked example's 0.47 GiB
    (which additionally counts the optical branch and attention). Without it the same
    pyramid gives 0.15 GiB — the SAR encoder alone.

    .. warning::

       **This is a lower bound, typically by a factor of 1.5--2.5.** The gap comes from
       allocator reservation (``max_memory_reserved`` exceeds ``max_memory_allocated``),
       cuDNN convolution workspace, the autocast casting cache, under-counting by the
       "4 tensors" approximation, and evaluation-time spikes. The 15 % criterion of
       ``P1-M4`` applies to the *calibrated* prediction
       :math:`k \cdot M_{\mathrm{analytic}}` (see :class:`MemoryBudget`).
    """
    if tensors_per_stage < 1 or bytes_per_element < 1:
        raise ValueError("tensors_per_stage and bytes_per_element must be >= 1")
    shapes = [tuple(int(s) for s in shape) for shape in stage_shapes]
    if any(len(shape) != 4 for shape in shapes):
        raise ValueError("every stage shape must be (B, C, H, W)")
    encoder_elements = sum(_numel(shape) for shape in shapes)
    decoder_elements = 2 * sum(_numel(shape) for shape in shapes[:-1]) if decoder_mirror else 0
    return (encoder_elements + decoder_elements) * tensors_per_stage * bytes_per_element


@contextmanager
def measure_peak(device: int = 0) -> Iterator[dict[str, int]]:
    """Context manager reporting peak allocated **and** reserved bytes.

    Yields
    ------
    dict[str, int]
        Populated on exit with ``allocated`` and ``reserved`` (peak values since entry)
        and ``cuda`` (1 if measured on a GPU, 0 otherwise).

    Notes
    -----
    Report **both**. A large gap between allocated and reserved indicates allocator
    fragmentation rather than genuine demand, which is a different problem with a
    different fix (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` rather than a
    smaller batch). Logging only ``allocated`` hides the distinction, and debugging
    challenge C2 in ``docs/10`` section 3 turns on exactly it.

    The values are the **peaks** (``max_memory_*``), read after
    ``torch.cuda.synchronize``: a tensor freed before the block exits still counts, and
    queued kernels have finished. On a machine without CUDA the dict holds zeros.
    """
    stats: dict[str, int] = {"allocated": 0, "reserved": 0, "cuda": 0}
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only on torch-free machines
        torch = None  # type: ignore[assignment]
    use_cuda = torch is not None and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield stats
    finally:
        if use_cuda:
            torch.cuda.synchronize(device)
            stats["allocated"] = int(torch.cuda.max_memory_allocated(device))
            stats["reserved"] = int(torch.cuda.max_memory_reserved(device))
            stats["cuda"] = 1


def collect_block_shapes(
    model: Any,
    example_inputs: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], ...]:
    """Record the output shape of every stored-activation block via forward hooks.

    One entry per ``Conv2d`` output (each conv opens a conv--norm--act block, the unit of
    the "4 tensors per block" rule) plus, for the fusion block, the token tensor of the
    attention. Modules flagged ``checkpointing_enabled`` contribute only their input
    (what ``torch.utils.checkpoint`` keeps); their internals are recomputed in backward
    and are **not** counted.

    Feed the result to :func:`estimate_activation_bytes` with ``decoder_mirror=False``.
    """
    import torch
    from torch import nn

    shapes: list[tuple[int, int, int, int]] = []
    checkpointed_prefixes: list[str] = [
        name for name, module in model.named_modules()
        if name and getattr(module, "checkpointing_enabled", False)
    ]

    def under_checkpoint(name: str) -> bool:
        return any(name.startswith(prefix + ".") for prefix in checkpointed_prefixes)

    def as_bchw(t: torch.Tensor) -> tuple[int, int, int, int] | None:
        if t.dim() == 4:
            return tuple(int(s) for s in t.shape)  # type: ignore[return-value]
        if t.dim() == 3:  # (B, N, C) tokens
            b, n, c = (int(s) for s in t.shape)
            return (b, c, n, 1)
        return None

    handles = []
    for name, module in model.named_modules():
        if name in checkpointed_prefixes:
            def ckpt_hook(_m, inputs, _out):
                first = inputs[0] if inputs else None
                if isinstance(first, torch.Tensor) and (s := as_bchw(first)) is not None:
                    shapes.append(s)
            handles.append(module.register_forward_hook(ckpt_hook))
        elif isinstance(module, nn.Conv2d | nn.Linear) and not under_checkpoint(name):
            def hook(_m, _inputs, out):
                if isinstance(out, torch.Tensor) and (s := as_bchw(out)) is not None:
                    shapes.append(s)
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(**example_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(shapes)


def load_overhead_factor(path: Path | str = Path("reports/vram_budget.json")) -> tuple[float, str]:
    """The calibrated overhead factor ``k`` from a budget run, or the default and why."""
    path = Path(path)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        k = float(rows[0]["overhead_factor_k"])
        return k, f"calibrated k from {path}"
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return DEFAULT_OVERHEAD_FACTOR, f"default k={DEFAULT_OVERHEAD_FACTOR} ({path} not available)"


def predict_training_peak(
    model: Any,
    example_inputs_b1: Mapping[str, Any],
    *,
    label: str,
    batch_size: int,
    chip_size: int,
    overhead_factor: float,
    bytes_per_element: int = 2,
) -> MemoryBudget:
    """Predict a run's training peak **before** its first step (``P1-M4``).

    The block shapes come from one batch-1 forward of the actual model (activations scale
    exactly linearly in B); static memory is 16 B/parameter for AdamW with fp32 master
    weights. ``overhead_factor`` is the ``k`` calibrated by ``make vram-budget``.
    """
    was_training = model.training
    model.eval()
    try:
        shapes = collect_block_shapes(model, example_inputs_b1)
    finally:
        model.train(was_training)
    analytic = estimate_activation_bytes(
        tuple((batch_size, c, h, w) for (_one, c, h, w) in shapes),
        tensors_per_stage=4, bytes_per_element=bytes_per_element, decoder_mirror=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return MemoryBudget(label=label, n_params=n_params, batch_size=batch_size, chip_size=chip_size,
                        static_bytes=static_bytes(n_params, "adamw"), activation_bytes_analytic=analytic,
                        overhead_factor=overhead_factor)


# ---------------------------------------------------------------------------
# Calibration CLI — docs/08 section 4.6
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetConfiguration:
    """One row of the docs/08 section 4.6 table."""

    label: str
    architecture: str
    batch_size: int
    chip_size: int
    checkpointing: bool = False


#: The configurations of docs/08 section 4.6, in that order. The first is the calibration
#: point; every other row is predicted *before* it is measured.
SECTION_4_6_CONFIGURATIONS: tuple[BudgetConfiguration, ...] = (
    BudgetConfiguration("Fusion, 256^2, B=8 (calibration point)", "fusion_crossattn", 8, 256),
    BudgetConfiguration("SAR-only U-Net, 256^2", "unet_sar", 24, 256),
    BudgetConfiguration("Fusion, 256^2", "fusion_crossattn", 16, 256),
    BudgetConfiguration("Fusion, 512^2", "fusion_crossattn", 4, 512),
    BudgetConfiguration("Fusion + checkpointing, 512^2", "fusion_crossattn", 16, 512, True),
)


def _budget_model(cfg: Any, conf: BudgetConfiguration, device: Any) -> Any:
    import torch

    from floodfusion.models.zoo import build_model

    model = build_model(replace(cfg.model, architecture=conf.architecture)).to(device)
    if conf.checkpointing:
        model.enable_gradient_checkpointing(True)
    return model.to(memory_format=torch.channels_last)


def _budget_batch(bsz: int, s: int, device: Any) -> dict[str, Any]:
    import torch

    from floodfusion.config import N_OPTICAL_CHANNELS, N_SAR_CHANNELS, N_TERRAIN_CHANNELS

    g = torch.Generator(device="cpu").manual_seed(0)
    out = {
        "sar": torch.randn(bsz, N_SAR_CHANNELS, s, s, generator=g),
        "terrain": torch.rand(bsz, N_TERRAIN_CHANNELS, s, s, generator=g),
        "optical": torch.rand(bsz, N_OPTICAL_CHANNELS, s, s, generator=g),
        "optical_quality": torch.ones(bsz, 1, s, s),
        "optical_available": torch.ones(bsz, dtype=torch.bool),
    }
    return {k: (v.to(device, memory_format=torch.channels_last) if v.dim() == 4 else v.to(device))
            for k, v in out.items()}


def _analytic_prediction(cfg: Any, conf: BudgetConfiguration, device: Any) -> tuple[int, int]:
    """(analytic activation bytes, n_params) from batch-1 block shapes — BEFORE any measurement.

    A batch-1 forward never runs out of memory, so every row gets its prediction even when the
    measured step then fails: "predicted 9 GiB, measured OOM" is a result; "predicted 0" is not.
    """
    import torch

    model = _budget_model(cfg, conf, device)
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()
    shapes = collect_block_shapes(model, _budget_batch(1, conf.chip_size, device))
    analytic = estimate_activation_bytes(
        tuple((conf.batch_size, c, h, w) for (_one, c, h, w) in shapes),
        tensors_per_stage=4, bytes_per_element=2, decoder_mirror=False,
    )
    del model
    torch.cuda.empty_cache()
    return analytic, n_params


def _training_step_peak(cfg: Any, conf: BudgetConfiguration, device: Any) -> tuple[int, int]:
    """Measure one bf16 training step: (max allocated, max reserved) bytes."""
    import torch

    from floodfusion.models.losses import CompoundFocalDiceLoss
    from floodfusion.optim import build_optimizer

    model = _budget_model(cfg, conf, device)
    model.train()
    b, s = conf.batch_size, conf.chip_size
    loss_fn = CompoundFocalDiceLoss(cfg.loss)
    optimizer = build_optimizer(model, cfg.train)
    inputs = _budget_batch(b, s, device)
    target = (torch.rand(b, 1, s, s, device=device) < 0.02).float()
    valid = torch.ones(b, 1, s, s, dtype=torch.bool, device=device)
    torch.cuda.empty_cache()
    try:
        with measure_peak() as stats:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _gate = model(**inputs)
                total, _parts = loss_fn(logits, target, valid)
            total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        del model, optimizer, inputs
        torch.cuda.empty_cache()
    return stats["allocated"], stats["reserved"]


def build_budget_table(cfg: Any, *, configurations: Sequence[BudgetConfiguration] = SECTION_4_6_CONFIGURATIONS) -> list[dict[str, Any]]:
    """Calibrate on the first configuration, then predict-then-measure every other one."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the VRAM budget table needs a CUDA device")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    rows: list[dict[str, Any]] = []
    k: float | None = None
    for conf in configurations:
        # The analytic estimate and the prediction are fixed BEFORE the measurement.
        analytic, n_params = _analytic_prediction(cfg, conf, device)
        try:
            allocated, reserved = _training_step_peak(cfg, conf, device)
            oom = False
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            allocated = reserved = 0
            oom = True
        static = static_bytes(n_params, "adamw")
        if k is None and not oom:
            k = max((allocated - static) / max(analytic, 1), 1.0)
        budget = MemoryBudget(
            label=conf.label, n_params=n_params, batch_size=conf.batch_size,
            chip_size=conf.chip_size, static_bytes=static,
            activation_bytes_analytic=analytic,
            overhead_factor=k if k is not None else DEFAULT_OVERHEAD_FACTOR,
            measured_bytes=None if oom else allocated,
        )
        rows.append({
            **asdict(conf),
            "n_params": n_params,
            "static_bytes": static,
            "activation_bytes_analytic": analytic,
            "overhead_factor_k": budget.overhead_factor,
            "predicted_bytes": budget.predicted_bytes,
            "measured_allocated_bytes": None if oom else allocated,
            "measured_reserved_bytes": None if oom else reserved,
            "relative_error": budget.relative_error,
            "oom": oom,
        })
        LOGGER.info("%s: predicted %.2f GiB, measured %s", conf.label,
                    budget.predicted_bytes / 1024**3,
                    "OOM" if oom else f"{allocated / 1024**3:.2f} GiB")
    return rows


def render_budget_markdown(rows: Sequence[Mapping[str, Any]], *, header: str = "") -> str:
    """docs/08 section 4.6 table as Markdown."""
    gib = 1024**3
    lines = [
        header,
        "| Configuration | P | B | Chip | Static | Activations (analytic) | k | Predicted peak | Measured (alloc / reserved) | Error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        measured = ("OOM" if r["oom"] else
                    f"{r['measured_allocated_bytes'] / gib:.2f} / {r['measured_reserved_bytes'] / gib:.2f} GiB")
        err = "—" if r["relative_error"] is None else f"{100 * r['relative_error']:+.1f} %"
        lines.append(
            f"| {r['label']} | {r['n_params'] / 1e6:.1f} M | {r['batch_size']} | {r['chip_size']} "
            f"| {r['static_bytes'] / gib:.2f} GiB | {r['activation_bytes_analytic'] / gib:.2f} GiB "
            f"| {r['overhead_factor_k']:.2f} | {r['predicted_bytes'] / gib:.2f} GiB | {measured} | {err} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m floodfusion.utils.vram --config configs/row09_fusion_crossattn.yaml``."""
    from floodfusion.config import load_config
    from floodfusion.utils.env import environment_record

    parser = argparse.ArgumentParser(description="P1-M4.1 VRAM budget: predict, measure, calibrate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/vram_budget.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    started = time.time()
    rows = build_budget_table(cfg)
    env = environment_record()
    header = (
        "# VRAM budget — docs/08 section 4.6 (measured)\n\n"
        f"Generated by `python -m floodfusion.utils.vram --config {args.config}` on "
        f"{env.get('gpu_name', 'unknown GPU')} (torch {env.get('torch')}, CUDA {env.get('cuda')}), "
        f"git `{env.get('git_sha', 'unknown')}`{' (dirty)' if env.get('git_dirty') else ''}.\n\n"
        "Method: analytic activations from forward-hook block shapes (4 stored tensors per block, "
        "bf16), static = 16 B/param (AdamW, fp32 master weights). Row 1 calibrates "
        "k = (measured − static) / analytic; every later row is **predicted with that k before it "
        "is measured**. Measured = `torch.cuda.max_memory_allocated()` over one optimiser step.\n"
    )
    markdown = render_budget_markdown(rows, header=header)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows, "environment": env}, indent=2, default=str), encoding="utf-8")
    worst = max((abs(r["relative_error"]) for r in rows[1:] if r["relative_error"] is not None), default=math.nan)
    LOGGER.info("wrote %s in %.0f s; worst predicted-row |error| = %.1f %%", args.out, time.time() - started, 100 * worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
