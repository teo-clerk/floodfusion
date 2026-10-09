r"""Training entry point for Project 01.

Milestone ``P1-M4``. Run as::

    python -m floodfusion.train --config 01_FLOOD_SAR_OPTICAL_FUSION/configs/row09_fusion_crossattn.yaml \
        --data-root data/synthetic --overlay configs/overlays/synthetic_demo.yaml --seed 42

(from inside the project directory the config path is ``configs/row09_fusion_crossattn.yaml``;
``make train-p1`` runs exactly that).

What a run does, in order (``docs/08`` section 7):

1. Resolve the frozen config (``extends`` chain, overlays, ``--set`` overrides) and refuse
   the rows that are not trainings: 1--3 (the physics baseline, ``baseline_runner``) and 11
   (row 9's checkpoints evaluated SAR-only, ``evaluate``).
2. Load the manifest, the frozen split **bound to that manifest's hash**, and the
   train-split normalisation statistics; refuse simulated pixels unless
   ``--allow-synthetic`` (``chip_source`` other than ``raster``).
3. Seed everything; build the model, the compound loss, AdamW (no decay on norm/bias/
   embedding) and the warmup + cosine schedule.
4. Write the VRAM prediction to ``run.json`` **before the first step**, plus git SHA,
   config hash, manifest hash, the training and deployment priors, and the environment.
5. Train with bf16 autocast, per-sample modality dropout, gradient accumulation and
   clipping; validate every epoch on per-basin confusion matrices; checkpoint on the best
   validation macro-over-basin IoU; stop early on validation only.

References
----------
``docs/08_training_recipe_and_vram.md`` sections 5, 7.
``docs/07_fusion_architectures.md`` section 7.3 — modality dropout belongs here.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from floodfusion.config import ExperimentConfig, config_sha256, config_to_dict, load_config
from floodfusion.data.dataset import (
    BalancedChipSampler,
    FloodChipDataset,
    build_dataloader,
    iter_batches_to_device,
    natural_prior,
)
from floodfusion.data.records import ChipRecord, chip_set_sha256, load_manifest, load_manifest_meta
from floodfusion.metrics import (
    ConfusionMatrix,
    accumulate_confusion,
    average_precision,
    macro_iou,
    precision_recall_curve,
    sum_confusions,
)
from floodfusion.models.losses import CompoundFocalDiceLoss
from floodfusion.models.zoo import build_model, count_parameters
from floodfusion.optim import apply_modality_dropout, build_optimizer, build_scheduler
from floodfusion.runs import append_metrics, run_dir_for, write_run_record
from floodfusion.splits import load_norm_stats, load_splits

__all__ = [
    "EVAL_CONTEXT_PX",
    "EmptySplitError",
    "TrainingData",
    "chip_set_sha256",
    "fit",
    "load_training_data",
    "main",
    "model_inputs",
    "train_one_epoch",
    "validate",
]

LOGGER = logging.getLogger(__name__)

#: Overlap-and-crop context for validation and test (``docs/05`` 8.1): predict on a
#: 256 + 2*64 = 384 px window, score the central 256 px. Removes zero-padding edge effects.
EVAL_CONTEXT_PX = 64
#: Row 6 is defined on cloud-free chips only (MILESTONES section 6).
ROW6_MIN_CLEAR_FRACTION = 0.9
#: Architectures with an optical input, to which modality dropout applies.
OPTICAL_FUSION_ARCHITECTURES = frozenset({"fusion_early", "fusion_late", "fusion_crossattn"})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class EmptySplitError(ValueError):
    """A split has no chips for this configuration (row 6 on cloudy data): the row is not run."""


@dataclass(frozen=True)
class TrainingData:
    """Everything a run reads from disk, with the hashes that bind it together."""

    train: tuple[ChipRecord, ...]
    val: tuple[ChipRecord, ...]
    test: tuple[ChipRecord, ...]
    cfg: ExperimentConfig
    manifest_sha256: str
    manifest_provenance: dict[str, Any]
    split_path: Path
    split_strategy: str
    norm_stats_path: Path
    split_sha256: str = ""

    @property
    def is_synthetic(self) -> bool:
        return "synthetic" in str(self.manifest_provenance.get("generator", ""))


def _filter_row6(records: Sequence[ChipRecord]) -> tuple[ChipRecord, ...]:
    return tuple(r for r in records if r.optical_available and r.optical_clear_fraction >= ROW6_MIN_CLEAR_FRACTION)


def load_training_data(cfg: ExperimentConfig, data_root: Path | str) -> TrainingData:
    """Manifest + frozen split (hash-checked) + train-only normalisation, per ``cfg``."""
    chip_dir = Path(data_root) / Path(cfg.data.chip_root).name
    manifest_path = chip_dir / cfg.data.manifest_name
    records = {r.chip_id: r for r in load_manifest(manifest_path)}
    meta = load_manifest_meta(manifest_path)
    split_path = chip_dir / cfg.data.split_file_name
    split = load_splits(split_path, expected_manifest_sha256=meta["manifest_sha256"])
    if split.strategy != cfg.data.split_strategy:
        raise ValueError(f"{split_path} has strategy {split.strategy!r}, config wants {cfg.data.split_strategy!r}")
    norm_path = chip_dir / cfg.data.norm_stats_file
    norm = load_norm_stats(norm_path, cfg.norm)
    cfg = replace(cfg, norm=norm)
    parts = [tuple(records[c] for c in split.chips(name)) for name in ("train", "val", "test")]
    if cfg.model.architecture == "unet_optical":
        parts = [_filter_row6(p) for p in parts]
    for name, part in zip(("train", "val", "test"), parts):
        if not part:
            raise EmptySplitError(f"the {name} split is empty for {cfg.name}")
    import hashlib

    return TrainingData(*parts, cfg=cfg, manifest_sha256=meta["manifest_sha256"],
                        manifest_provenance=meta.get("provenance", {}), split_path=split_path,
                        split_strategy=split.strategy, norm_stats_path=norm_path,
                        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def model_inputs(batch: Mapping[str, Any], available: torch.Tensor, *, force_sar_only: bool = False) -> dict[str, Any]:
    """The keyword arguments every architecture's ``forward`` accepts."""
    if force_sar_only:
        return {"sar": batch["sar"], "terrain": batch["terrain"], "optical": None,
                "optical_quality": None, "optical_available": torch.zeros_like(available)}
    return {"sar": batch["sar"], "terrain": batch["terrain"], "optical": batch["optical"],
            "optical_quality": batch["optical_quality"], "optical_available": available}


def _autocast(device: torch.device, cfg: ExperimentConfig) -> Any:
    enabled = device.type == "cuda" and cfg.train.amp_dtype == "bfloat16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _gate_stats(gate: torch.Tensor | None, available: torch.Tensor, sums: dict[str, float]) -> None:
    if gate is None:
        return
    per_sample = gate.detach().float().mean(dim=tuple(range(1, gate.dim())))
    for key, mask in (("gate", torch.ones_like(available)), ("gate_with_optical", available),
                      ("gate_without_optical", ~available)):
        if bool(mask.any()):
            sums[key] += float(per_sample[mask].sum())
            sums[f"n_{key}"] += int(mask.sum())


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    cfg: ExperimentConfig,
    dropout_generator: torch.Generator,
) -> dict[str, float]:
    r"""Run one training epoch (``docs/08`` section 7.4).

    Returns ``loss``, ``loss_focal``, ``loss_dice``, ``gate_mean`` (+ split by optical
    availability), ``lr``, ``grad_norm``, ``peak_vram_bytes``, ``peak_reserved_bytes``,
    ``seconds``, ``optimizer_steps`` and ``dropped_fraction``.

    The traps, handled: the loss is **divided by** ``accum_steps``; modality dropout is
    drawn **per sample** and never touches SAR; the micro-batches of an epoch are truncated
    to a whole number of optimiser steps, so every step averages exactly ``accum_steps``
    micro-batches and the scheduler sees one step per optimiser step. Gradient accumulation
    is exact for focal and *not* for Dice (non-decomposable; ``docs/08`` section 5.3), which
    is why the largest micro-batch that fits is preferred. A non-finite loss raises.
    """
    accum = cfg.train.grad_accum_steps
    n_micro = (len(loader) // accum) * accum
    if n_micro == 0:
        raise ValueError(f"{len(loader)} micro-batches cannot fill one step of {accum}")
    p_drop = cfg.model.modality_dropout_p if cfg.model.architecture in OPTICAL_FUSION_ARCHITECTURES else 0.0
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sums: dict[str, float] = defaultdict(float)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(iter_batches_to_device(loader, device, channels_last=cfg.train.channels_last)):
        if i >= n_micro:
            break
        available = apply_modality_dropout(batch["optical_available"], p_drop, generator=dropout_generator)
        sums["dropped"] += float((batch["optical_available"] & ~available).sum())
        sums["n_samples"] += available.numel()
        with _autocast(device, cfg):
            logits, gate = model(**model_inputs(batch, available))
        total, parts = loss_fn(logits.float(), batch["label"], batch["valid"])
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss at micro-batch {i}: focal={float(parts['focal'])}, "
                                     f"dice={float(parts['dice'])}, chips={list(batch['chip_id'])[:4]}")
        (total / accum).backward()
        sums["loss"] += float(total.detach())
        sums["loss_focal"] += float(parts["focal"])
        sums["loss_dice"] += float(parts["dice"])
        _gate_stats(gate, available, sums)
        if (i + 1) % accum == 0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at micro-batch {i} "
                                         f"(chips {list(batch['chip_id'])[:4]})")
            sums["grad_norm"] += grad_norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            sums["steps"] += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    steps = max(sums["steps"], 1.0)

    def mean_gate(key: str) -> float:
        return sums[key] / sums[f"n_{key}"] if sums[f"n_{key}"] else float("nan")

    return {
        "loss": sums["loss"] / n_micro, "loss_focal": sums["loss_focal"] / n_micro,
        "loss_dice": sums["loss_dice"] / n_micro, "gate_mean": mean_gate("gate"),
        "gate_mean_with_optical": mean_gate("gate_with_optical"),
        "gate_mean_without_optical": mean_gate("gate_without_optical"),
        "lr": float(optimizer.param_groups[0]["lr"]), "grad_norm": sums["grad_norm"] / steps,
        "optimizer_steps": int(sums["steps"]),
        "dropped_fraction": sums["dropped"] / max(sums["n_samples"], 1.0),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "seconds": time.perf_counter() - started,
    }


def _crop(t: torch.Tensor, c: int) -> torch.Tensor:
    return t[..., c:t.shape[-2] - c, c:t.shape[-1] - c] if c else t


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    cfg: ExperimentConfig,
    thresholds: Sequence[float] | None = None,
    force_sar_only: bool = False,
    crop_px: int = 0,
    logit_sink: Any = None,
    temperature: float = 1.0,
    offset: float = 0.0,
    per_chip_threshold: float | None = None,
) -> dict[str, Any]:
    r"""Per-basin confusion matrices at fixed thresholds; macro-over-basin IoU.

    Runs under ``inference_mode`` and accumulates **counts**, never probability maps — the
    two commonest causes of the validation OOM (``docs/08`` section 7.3). With
    ``crop_px`` the loader's context border is predicted but not scored. ``logit_sink``
    (``LogitReservoir``-like) receives the valid-pixel logits and labels for calibration.

    ``thresholds`` are on the **calibrated** probability scale
    :math:`p = \sigma((z + c)/T)`; they are compared in logit space,
    :math:`p > \tau \iff z > T\operatorname{logit}(\tau) - c`, which is exact and avoids
    the float32 saturation of a sigmoid near 0 and 1. ``T = 1, c = 0`` is the raw score.

    Returns ``per_basin`` (basin -> one :class:`ConfusionMatrix` per threshold),
    ``macro_iou_by_threshold``, ``micro_iou_by_threshold``, ``best_threshold``,
    ``best_macro_iou`` and the pooled ``average_precision``. With ``per_chip_threshold``
    (calibrated scale) also ``per_chip``: chip id -> ``[tp, fp, fn, tn]`` at that threshold,
    so any subset of the chips can be re-scored without running the model again.
    """
    thresholds = tuple(cfg.eval.pr_thresholds if thresholds is None else thresholds)
    if not all(0.0 < t < 1.0 for t in thresholds):
        raise ValueError("thresholds must lie strictly inside (0, 1)")
    score_thresholds = tuple(temperature * math.log(t / (1.0 - t)) - offset for t in thresholds)
    model.eval()
    per_basin: dict[str, list[ConfusionMatrix]] = {}
    per_chip: dict[str, list[int]] = {}
    chip_score = (None if per_chip_threshold is None
                  else temperature * math.log(per_chip_threshold / (1.0 - per_chip_threshold)) - offset)
    for batch in iter_batches_to_device(loader, device, channels_last=cfg.train.channels_last):
        with _autocast(device, cfg):
            logits, _ = model(**model_inputs(batch, batch["optical_available"], force_sar_only=force_sar_only))
        logits = _crop(logits.float(), crop_px)
        label, valid = _crop(batch["label"], crop_px), _crop(batch["valid"], crop_px)
        if not bool(torch.isfinite(logits[valid]).all()):
            # NaN sorts last in searchsorted and would be counted as flood: fail instead.
            raise FloatingPointError(f"non-finite logits on valid pixels (chips {list(batch['chip_id'])[:4]})")
        basins = list(batch["basin_id"])
        for basin in dict.fromkeys(basins):
            idx = torch.tensor([j for j, b in enumerate(basins) if b == basin], device=logits.device)
            cms = accumulate_confusion(logits.index_select(0, idx), label.index_select(0, idx),
                                       valid.index_select(0, idx), score_thresholds)
            current = per_basin.get(basin)
            per_basin[basin] = list(cms) if current is None else [a + b for a, b in zip(current, cms)]
        if chip_score is not None:
            for j, chip in enumerate(batch["chip_id"]):
                (cm,) = accumulate_confusion(logits[j:j + 1], label[j:j + 1], valid[j:j + 1], (chip_score,))
                per_chip[str(chip)] = [cm.tp, cm.fp, cm.fn, cm.tn]
        if logit_sink is not None:
            logit_sink.add(logits[valid].cpu().numpy(), (label[valid] > 0.5).cpu().numpy())
    if not per_basin:
        raise ValueError("validation loader yielded no batches")
    frozen = {b: tuple(v) for b, v in per_basin.items()}
    macro = [macro_iou({b: v[k] for b, v in frozen.items()}) for k in range(len(thresholds))]
    pooled = [sum_confusions([v[k] for v in frozen.values()]) for k in range(len(thresholds))]
    finite = [m if math.isfinite(m) else -1.0 for m in macro]
    best = int(np.argmax(finite))
    try:
        ap = average_precision(*precision_recall_curve(tuple(pooled)))
    except ValueError:
        ap = float("nan")  # no positive pixel in the split
    return {
        "thresholds": thresholds, "per_basin": frozen, "macro_iou_by_threshold": macro,
        "micro_iou_by_threshold": [m.iou for m in pooled], "best_threshold": thresholds[best],
        "best_macro_iou": macro[best], "average_precision": ap,
        **({"per_chip": per_chip} if chip_score is not None else {}),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (``docs/08`` 7.2); optionally deterministic kernels.

    With ``deterministic``: ``use_deterministic_algorithms(True, warn_only=True)`` —
    deterministic implementations where they exist, a once-per-op warning where they do not.
    Off by default (``train.deterministic``): it costs 2.9x throughput here, and the
    residual non-determinism is measured by running three seeds instead.
    """
    import warnings

    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - the legacy global RNG third-party code uses
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS GEMMs
        torch.use_deterministic_algorithms(True, warn_only=True)
        warnings.filterwarnings("once", message=".*deterministic.*")
    else:
        torch.use_deterministic_algorithms(False)


def configure_torch(cfg: ExperimentConfig, device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.train.cudnn_benchmark)  # fixed chip shapes only


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _refuse_untrainable(cfg: ExperimentConfig) -> None:
    if cfg.model.architecture == "threshold_baseline":
        raise SystemExit(f"{cfg.name}: rows 1-3 are not trained — run `python -m floodfusion.baseline_runner`")
    if cfg.eval.force_sar_only:
        raise SystemExit(f"{cfg.name}: row 11 evaluates row 9's checkpoints SAR-only — run `python -m "
                         "floodfusion.evaluate --run-dir <row09 run> --force-sar-only "
                         "--out-dir runs/row11_fusion_eval_sar_only/seed<N>`")


def _example_inputs(cfg: ExperimentConfig, device: torch.device, batch_size: int = 1) -> dict[str, torch.Tensor]:
    from floodfusion.config import N_OPTICAL_CHANNELS, N_SAR_CHANNELS, N_TERRAIN_CHANNELS

    s = cfg.data.chip_size
    return {
        "sar": torch.zeros(batch_size, N_SAR_CHANNELS, s, s, device=device),
        "terrain": torch.zeros(batch_size, N_TERRAIN_CHANNELS, s, s, device=device),
        "optical": torch.zeros(batch_size, N_OPTICAL_CHANNELS, s, s, device=device),
        "optical_quality": torch.ones(batch_size, 1, s, s, device=device),
        "optical_available": torch.ones(batch_size, dtype=torch.bool, device=device),
    }


def _vram_prediction(model: nn.Module, cfg: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    from floodfusion.utils.vram import load_overhead_factor, predict_training_peak

    k, source = load_overhead_factor()
    budget = predict_training_peak(
        model, _example_inputs(cfg, device), label=cfg.name, batch_size=cfg.train.micro_batch_size,
        chip_size=cfg.data.chip_size, overhead_factor=k,
        bytes_per_element=2 if device.type == "cuda" and cfg.train.amp_dtype == "bfloat16" else 4,
    )
    return {"predicted_bytes": budget.predicted_bytes, "static_bytes": budget.static_bytes,
            "activation_bytes_analytic": budget.activation_bytes_analytic, "overhead_factor_k": k,
            "k_source": source, "micro_batch_size": cfg.train.micro_batch_size}


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    """``torch.save`` to a temporary name, then rename: never a half-written checkpoint."""
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _loaders(data: TrainingData, cfg: ExperimentConfig) -> tuple[Any, Any, BalancedChipSampler]:
    train_ds = FloodChipDataset(data.train, cfg.norm, cfg.data, cfg.grid, training=True)
    val_ds = FloodChipDataset(data.val, cfg.norm, cfg.data, cfg.grid, context_px=EVAL_CONTEXT_PX)
    sampler = BalancedChipSampler(data.train, negative_to_positive_ratio=cfg.data.negative_to_positive_ratio,
                                  positive_fraction_threshold=cfg.data.positive_fraction_threshold,
                                  seed=cfg.train.seed)
    train_loader = build_dataloader(train_ds, cfg.data, batch_size=cfg.train.micro_batch_size, shuffle=False,
                                    drop_last=True, sampler=sampler, seed=cfg.train.seed)
    val_loader = build_dataloader(val_ds, cfg.data, batch_size=cfg.train.micro_batch_size, shuffle=False,
                                  drop_last=False, seed=cfg.train.seed)
    return train_loader, val_loader, sampler


def _without_seed(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**config, "train": {k: v for k, v in config["train"].items() if k != "seed"}}


def _run_record(cfg: ExperimentConfig, data: TrainingData, model: nn.Module, sampler: BalancedChipSampler,
                device: torch.device, provenance: Mapping[str, Any]) -> dict[str, Any]:
    from floodfusion.utils.env import environment_record, git_state

    return {
        "name": cfg.name, "ablation_row": cfg.ablation_row, "architecture": cfg.model.architecture,
        "seed": cfg.train.seed, "config": config_to_dict(cfg), "config_sha256": config_sha256(config_to_dict(cfg)),
        "config_sha256_without_seed": config_sha256(_without_seed(config_to_dict(cfg))),
        "manifest_sha256": data.manifest_sha256, "synthetic_data": data.is_synthetic,
        "split_file": str(data.split_path), "split_strategy": data.split_strategy,
        "split_sha256": data.split_sha256, "val_chips_sha256": chip_set_sha256(data.val),
        "test_chips_sha256": chip_set_sha256(data.test),
        "norm_stats_file": str(data.norm_stats_path),
        "n_chips": {"train": len(data.train), "val": len(data.val), "test": len(data.test)},
        "pi_train": sampler.training_prior(), "pi_deploy": natural_prior(data.val),
        "n_parameters": count_parameters(model), "vram_prediction": _vram_prediction(model, cfg, device),
        "device": str(device), "environment": environment_record(), **git_state(), **dict(provenance),
        "started_unix": time.time(), "status": "running",
    }


@dataclass
class _Progress:
    start_epoch: int = 0
    best: float = -math.inf
    best_epoch: int = -1
    epochs_run: int = 0
    peak_vram_bytes: int = 0
    elapsed_seconds: float = 0.0  # training time of earlier sessions (resume)


def _restore(path: Path, model: nn.Module, optimizer: Any, scheduler: Any, generator: torch.Generator,
             device: torch.device, config_sha256_expected: str) -> _Progress:
    """Continue from ``last.pt`` — only if it was written by this exact configuration."""
    state = torch.load(path, map_location=device, weights_only=True)  # never unpickle arbitrary code
    if state.get("config_sha256") != config_sha256_expected:
        raise ValueError(f"{path} was written by a different configuration; refusing to resume")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    generator.set_state(state["dropout_generator"].cpu())  # Generator state must live on the CPU
    LOGGER.info("resumed from %s (epoch %d, best %.4f)", path, state["epoch"], state["best"])
    return _Progress(start_epoch=state["epoch"] + 1, best=state["best"], best_epoch=state["best_epoch"],
                     epochs_run=state["epoch"] + 1, peak_vram_bytes=int(state.get("peak_vram_bytes", 0)),
                     elapsed_seconds=float(state.get("elapsed_seconds", 0.0)))


def _log_epoch(cfg: ExperimentConfig, epoch: int, stats: Mapping[str, float], val: Mapping[str, Any]) -> dict[str, Any]:
    best_index = list(val["thresholds"]).index(val["best_threshold"])
    LOGGER.info("%s seed %d epoch %d: loss %.4f (focal %.4f, dice %.4f) gate %.3f/%.3f | val macro IoU %.3f "
                "@%.3f AP %.3f | lr %.2e | %.2f GiB | %.1f s", cfg.name, cfg.train.seed, epoch, stats["loss"],
                stats["loss_focal"], stats["loss_dice"], stats["gate_mean_with_optical"],
                stats["gate_mean_without_optical"], val["best_macro_iou"], val["best_threshold"],
                val["average_precision"], stats["lr"], stats["peak_vram_bytes"] / 1024**3, stats["seconds"])
    return {"epoch": epoch, **stats, "val_best_macro_iou": val["best_macro_iou"],
            "val_best_threshold": val["best_threshold"], "val_ap": val["average_precision"],
            "val_micro_iou_at_best": val["micro_iou_by_threshold"][best_index]}


def fit(
    data: TrainingData,
    run_dir: Path,
    *,
    device: torch.device,
    resume: Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train ``data.cfg`` (one configuration, one seed); returns the final ``run.json``."""
    cfg = data.cfg
    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)
    configure_torch(cfg, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.model).to(device)
    if cfg.train.grad_checkpointing:
        model.enable_gradient_checkpointing(True)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    train_loader, val_loader, sampler = _loaders(data, cfg)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train, len(train_loader) // cfg.train.grad_accum_steps)
    loss_fn = CompoundFocalDiceLoss(cfg.loss)
    generator = torch.Generator().manual_seed(cfg.train.seed + 1)
    record = _run_record(cfg, data, model, sampler, device, provenance or {})
    if resume is None:
        (run_dir / "metrics.jsonl").unlink(missing_ok=True)  # a fresh run never appends to a stale log
    write_run_record(run_dir, record)  # the VRAM prediction is on disk BEFORE step 1
    progress = (_restore(resume, model, optimizer, scheduler, generator, device, record["config_sha256"])
                if resume else _Progress())
    started = time.perf_counter()
    for epoch in range(progress.start_epoch, cfg.train.epochs):
        sampler.set_epoch(epoch)
        stats = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device=device, cfg=cfg,
                                dropout_generator=generator)
        val = validate(model, val_loader, device=device, cfg=cfg, crop_px=EVAL_CONTEXT_PX)
        progress.epochs_run = epoch + 1
        progress.peak_vram_bytes = max(progress.peak_vram_bytes, stats["peak_vram_bytes"])
        if math.isfinite(val["best_macro_iou"]) and val["best_macro_iou"] > progress.best:
            progress.best, progress.best_epoch = val["best_macro_iou"], epoch
            _save(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch, "val_macro_iou": progress.best,
                                        "config_sha256": record["config_sha256"]})
        _save(run_dir / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best": progress.best,
                                    "best_epoch": progress.best_epoch, "dropout_generator": generator.get_state(),
                                    "config_sha256": record["config_sha256"],
                                    "peak_vram_bytes": progress.peak_vram_bytes,
                                    "elapsed_seconds": progress.elapsed_seconds + time.perf_counter() - started})
        append_metrics(run_dir, _log_epoch(cfg, epoch, stats, val))
        patience = cfg.train.early_stopping_patience
        if patience and epoch - progress.best_epoch >= patience:
            LOGGER.info("early stop at epoch %d: no improvement since epoch %d", epoch, progress.best_epoch)
            break
    peak = progress.peak_vram_bytes
    predicted = record["vram_prediction"]["predicted_bytes"]
    final = {
        **record, "status": "finished", "best_epoch": progress.best_epoch, "best_val_macro_iou": progress.best,
        "epochs_run": progress.epochs_run,
        "train_seconds": progress.elapsed_seconds + time.perf_counter() - started,
        "peak_vram_bytes": peak, "finished_unix": time.time(),
        "vram_prediction": {**record["vram_prediction"],
                            "relative_error": (predicted - peak) / peak if peak else None},
    }
    write_run_record(run_dir, final)
    del model, optimizer, scheduler, train_loader, val_loader
    if device.type == "cuda":
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return final


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description="Train the Project 01 flood segmenter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="a last.pt to continue from")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="a.b=value override")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="permit chip_source other than 'raster' (in-memory simulated pixels)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = list(args.overrides) + ([f"train.seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.config, overlays=args.overlay, overrides=overrides)
    _refuse_untrainable(cfg)
    if cfg.data.chip_source != "raster" and not args.allow_synthetic:
        raise SystemExit(f"chip_source={cfg.data.chip_source!r} may substitute simulated pixels; "
                         "pass --allow-synthetic to accept that")
    data = load_training_data(cfg, args.data_root)
    run_dir = args.run_dir or run_dir_for(args.runs_root, cfg.name, cfg.train.seed)
    fit(data, run_dir, device=resolve_device(args.device), resume=args.resume, provenance={
        "config_path": str(args.config), "overlays": [str(o) for o in args.overlay],
        "overrides": overrides, "data_root": str(args.data_root),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
