"""The fair TerraMind's size and adaptation, by the rule fixed on 2026-10-13 (reports/v1_2_deviations.md).

    make tm-probe      # through scripts/v12_queue.sh, which holds the GPU lock; ~10 min; writes reports/v12_tm_probe.json

Run it only through the queue: a second GPU job running at the same time would distort both timings and memory peaks.

**What is timed.** Row 9 and each TerraMind-fair variant (base + LoRA, base frozen, small + LoRA) train on the **same**
micro-batches: a seeded sample of chips from across the manifest, loaded once. Each step is the training loop's own
(``train.train_one_epoch``):
- modality dropout at the config's rate;
- bf16 autocast, loss / accumulation steps, backward;
- every ``grad_accum_steps`` micro-batches: gradient clipping, the project's optimizer step (``build_optimizer``);
- ``channels_last`` model and inputs, ``configure_torch``.

**The projection.** A pooled cross-validation run is projected as:
row 9's mean v1.1 Kuro Siwo run (:data:`ROW9_KS_RUN_S`) × pooled training chips / Kuro Siwo's × the variant's time per
micro-batch / row 9's. This assumes the same number of epochs (early stopping may differ), and it scales validation
and data loading by the same ratio, because :data:`ROW9_KS_RUN_S` is a whole-run time.

**The memory cap.** The peak **reserved** memory (what the allocator holds, not only live tensors) must be at most
:data:`VRAM_CAP_GB`.

**The rule.** Among the variants within the memory cap, the first in the order above whose projection is at most
:data:`CAP_H` hours is the fair opponent; if none is that fast, the last of them (the cheapest). If none fits in
memory, there is no fair opponent: the probe exits 1 and the CV stage refuses to start.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAP_H = 3.0
VRAM_CAP_GB = 7.5
#: v1.1: row 9 on Kuro Siwo (all-groups and folds), mean train_seconds.
ROW9_KS_RUN_S = 972.0
#: v1.1 Kuro Siwo training chips per fold (0.8 x 3 971).
KS_TRAIN_CHIPS = 3178
VARIANTS = ("terramind_base_fair", "terramind_base_fair_frozen", "terramind_small_fair")
#: Micro-batches: warm-up, then timed. Both are whole optimizer steps (multiples of grad_accum_steps = 4).
WARMUP, TIMED = 8, 32
SAMPLE_SEED = 0


def _config(path: Path) -> Any:
    from floodfusion.config import load_config

    return load_config(path, overlays=[ROOT / "configs/overlays/kurosiwo.yaml"])


def sample_batches(cfg: Any, data_root: Path, n: int) -> list[dict[str, Any]]:
    """``n`` CPU micro-batches of seeded random chips from across the manifest, with the DEM sidecar read."""
    import numpy as np

    from floodfusion.data.dataset import FloodChipDataset, build_dataloader
    from floodfusion.data.records import load_manifest

    records = load_manifest(data_root / "chips" / "manifest.parquet")
    order = np.random.default_rng(SAMPLE_SEED).permutation(len(records))
    picked = [records[i] for i in order[: n * cfg.train.micro_batch_size]]
    ds = FloodChipDataset(picked, cfg.norm, cfg.data, cfg.grid, training=True, sidecars=("dem",))
    loader = build_dataloader(ds, cfg.data, batch_size=cfg.train.micro_batch_size, shuffle=False, drop_last=True)
    return [dict(b) for b in loader][:n]


def time_model(config: Path, batches: list[dict[str, Any]]) -> dict[str, float]:
    import torch
    from torch import nn

    from floodfusion.data.sidecars import sidecars_for
    from floodfusion.models.losses import CompoundFocalDiceLoss
    from floodfusion.models.zoo import build_model
    from floodfusion.optim import apply_modality_dropout, build_optimizer
    from floodfusion.train import (
        OPTICAL_FUSION_ARCHITECTURES,
        _autocast,
        configure_torch,
        model_inputs,
    )

    cfg = _config(config)
    device = torch.device("cuda")
    configure_torch(cfg, device)
    model = build_model(cfg.model).to(device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    optimizer = build_optimizer(model, cfg.train)
    loss_fn = CompoundFocalDiceLoss(cfg.loss)
    accum = cfg.train.grad_accum_steps
    p_drop = cfg.model.modality_dropout_p if cfg.model.architecture in OPTICAL_FUSION_ARCHITECTURES else 0.0
    generator = torch.Generator().manual_seed(cfg.train.seed + 1)
    keep_context = bool(sidecars_for(cfg.model.architecture))

    def to_device(batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in batch.items():
            if key.startswith("context") and not keep_context:
                continue
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                if cfg.train.channels_last and value.dim() == 4 and value.is_floating_point():
                    value = value.contiguous(memory_format=torch.channels_last)
            out[key] = value
        return out

    def step(i: int, batch: dict[str, Any]) -> None:
        batch = to_device(batch)
        available = apply_modality_dropout(batch["optical_available"], p_drop, generator=generator)
        with _autocast(device, cfg):
            logits, _ = model(**model_inputs(batch, available))
        total, _ = loss_fn(logits.float(), batch["label"], batch["valid"])
        (total / accum).backward()
        if (i + 1) % accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    for i, b in enumerate(batches[:WARMUP]):
        step(i, b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i, b in enumerate(batches[WARMUP:WARMUP + TIMED]):
        step(i, b)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / TIMED
    return {"s_per_micro_batch": seconds, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def choose(timings: dict[str, dict[str, float]], pool_chips: int) -> tuple[str | None, float]:
    """The pre-registered rule (module docstring): the chosen variant (or None) and row 9's projected run in hours."""
    base_run_h = ROW9_KS_RUN_S * pool_chips / KS_TRAIN_CHIPS / 3600
    for t in timings.values():
        t["projected_run_h"] = base_run_h * t["s_per_micro_batch"] / timings["row09"]["s_per_micro_batch"]
    fits = [v for v in VARIANTS if timings[v]["peak_reserved_gb"] <= VRAM_CAP_GB]
    chosen = next((v for v in fits if timings[v]["projected_run_h"] <= CAP_H), fits[-1] if fits else None)
    return chosen, base_run_h


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kurosiwo")
    parser.add_argument("--pool-chips", type=int, required=True, help="training chips of a pooled CV fold")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v12_tm_probe.json")
    args = parser.parse_args(argv)
    configs = {"row09": ROOT / "configs/row09_fusion_crossattn.yaml",
               **{v: ROOT / f"configs/comparisons/{v}.yaml" for v in VARIANTS}}
    batches = sample_batches(_config(configs[VARIANTS[0]]), args.data_root, WARMUP + TIMED)
    timings = {}
    for name, path in configs.items():
        timings[name] = time_model(path, batches)
        gc.collect()
        torch.cuda.empty_cache()   # each model's peak is measured alone
    chosen, base_run_h = choose(timings, args.pool_chips)
    report = {"rule": f"first of {list(VARIANTS)} with peak reserved <= {VRAM_CAP_GB} GB and projected pooled run "
                      f"<= {CAP_H} h; else the last that fits in memory; none fits: no fair opponent",
              "assumptions": "same epoch count as row 9; validation and loading scale with the step-time ratio",
              "pool_chips": args.pool_chips, "row9_projected_run_h": base_run_h, "timed_micro_batches": TIMED,
              "timings": timings, "chosen": chosen}
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
