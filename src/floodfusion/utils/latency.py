r"""Inference latency, throughput and peak VRAM, measured properly (``docs/09`` section 7).

=================== =====================================================================
quantity            how
=================== =====================================================================
thermal warm-up     sustained batched inference for ``warmup_s`` seconds first: at 45 W the
                    clocks fall 15--25 % below the short-burst figure once the card is hot
latency             batch 1, one 256^2 chip, >= 20 warm-up iterations, ``synchronize()``
                    around every timed call; median and p90 over ``n_timed`` calls
throughput          the largest batch that fits, chips/s sustained over ``throughput_s``
peak VRAM           ``max_memory_allocated()`` of a batch-1 call under ``inference_mode``
=================== =====================================================================

Converted into the operational unit: a 256^2 chip at 10 m covers 6.55 km^2, an IW frame
~42 500 km^2, hence ~6 500 chips per frame. Inference is not the bottleneck of the
``docs/00`` section 5.1 latency budget — the acquisition wait (0--24 h) and pre-processing
(0.5--2 h) are — and the report should say so.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn

from floodfusion.config import N_OPTICAL_CHANNELS, N_SAR_CHANNELS, N_TERRAIN_CHANNELS

__all__ = ["CHIPS_PER_IW_FRAME", "measure_inference"]

#: 250 km x 170 km / (2.56 km)^2.
CHIPS_PER_IW_FRAME = round(250 * 170 / 2.56**2)
_THROUGHPUT_BATCHES = (64, 32, 16, 8, 4, 2, 1)


def _inputs(batch: int, chip: int, device: torch.device, *, sar_only: bool) -> dict[str, Any]:
    g = torch.Generator(device="cpu").manual_seed(0)
    out: dict[str, Any] = {
        "sar": torch.randn(batch, N_SAR_CHANNELS, chip, chip, generator=g),
        "terrain": torch.rand(batch, N_TERRAIN_CHANNELS, chip, chip, generator=g),
        "optical": None if sar_only else torch.rand(batch, N_OPTICAL_CHANNELS, chip, chip, generator=g),
        "optical_quality": None if sar_only else torch.ones(batch, 1, chip, chip),
        "optical_available": torch.full((batch,), not sar_only, dtype=torch.bool),
    }
    return {k: (v.to(device, memory_format=torch.channels_last) if isinstance(v, torch.Tensor) and v.dim() == 4
                else v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


def _largest_batch(model: nn.Module, chip: int, device: torch.device, *, sar_only: bool, autocast: Any) -> int:
    for batch in _THROUGHPUT_BATCHES:
        try:
            with autocast():
                model(**_inputs(batch, chip, device, sar_only=sar_only))
            torch.cuda.synchronize(device)
            return batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("not even batch 1 fits for inference")


@torch.inference_mode()
def measure_inference(
    model: nn.Module,
    device: torch.device,
    *,
    chip_size: int = 256,
    sar_only: bool = False,
    warmup_s: float = 30.0,
    warmup_iters: int = 20,
    n_timed: int = 100,
    throughput_s: float = 60.0,
    bf16: bool = True,
) -> dict[str, Any]:
    """Latency / throughput / peak inference VRAM on a CUDA device (``docs/09`` section 7)."""
    if device.type != "cuda":
        raise RuntimeError("latency is defined on the deployment GPU; got a non-CUDA device")
    model.eval()

    def autocast() -> Any:
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)

    # Peak inference memory FIRST, with nothing but the model and one chip on the device.
    one = _inputs(1, chip_size, device, sar_only=sar_only)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    with autocast():
        model(**one)
    torch.cuda.synchronize(device)
    peak_b1 = torch.cuda.max_memory_allocated(device)

    batch = _largest_batch(model, chip_size, device, sar_only=sar_only, autocast=autocast)
    big = _inputs(batch, chip_size, device, sar_only=sar_only)
    started = time.perf_counter()
    while time.perf_counter() - started < warmup_s:  # thermally warm, not short-burst clocks
        with autocast():
            model(**big)
    torch.cuda.synchronize(device)

    for _ in range(warmup_iters):
        with autocast():
            model(**one)
    timings = []
    for _ in range(n_timed):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with autocast():
            model(**one)
        torch.cuda.synchronize(device)
        timings.append(1e3 * (time.perf_counter() - t0))


    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < throughput_s:
        with autocast():
            model(**big)
        n += batch
    torch.cuda.synchronize(device)
    chips_per_s = n / (time.perf_counter() - t0)
    median_ms = statistics.median(timings)
    return {
        "latency_ms_median": median_ms,
        "latency_ms_p90": statistics.quantiles(timings, n=10)[-1],
        "throughput_chips_per_s": chips_per_s, "throughput_batch": batch,
        "peak_inference_vram_bytes_b1": int(peak_b1),  # weights included: what deployment needs
        "peak_inference_activation_bytes_b1": int(peak_b1 - base), "sar_only": sar_only,
        "warmup_s": warmup_s, "throughput_s": throughput_s, "gpu": torch.cuda.get_device_name(device),
        "chips_per_iw_frame": CHIPS_PER_IW_FRAME,
        "minutes_per_iw_frame_batch1": CHIPS_PER_IW_FRAME * median_ms / 1e3 / 60.0,
        "minutes_per_iw_frame_throughput": CHIPS_PER_IW_FRAME / chips_per_s / 60.0,
    }
