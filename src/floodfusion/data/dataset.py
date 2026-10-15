"""PyTorch ``Dataset`` and ``DataLoader`` for co-registered flood chips.

This is the module in which most of Project 01's silent bugs are possible, so read
``tests/test_dataset.py`` before writing a line: it is the specification.

The contract
------------
Every sample is a ``dict`` with fixed keys, dtypes and shapes. ``H = W = chip_size``
(plus ``2 * context_px`` when evaluation reads context for overlap-and-crop).

============================ ==================== ============ ==============================
key                          shape                dtype        meaning
============================ ==================== ============ ==============================
``sar``                      ``(5, H, W)``        ``float32``  normalised dB, always present
``optical``                  ``(4, H, W)``        ``float32``  normalised reflectance + MNDWI
``terrain``                  ``(3, H, W)``        ``float32``  slope, log1p(HAND), theta_loc
``label``                    ``(1, H, W)``        ``float32``  flood mask in {0, 1}
``valid``                    ``(1, H, W)``        ``bool``     True where scored
``optical_quality``          ``(1, H, W)``        ``float32``  in [0, 1]; drives the gate
``optical_available``        ``()``               ``bool``     scalar; False -> use embedding
``landcover``                ``(1, H, W)``        ``uint8``    WorldCover code (analysis only)
``synthetic``                ``()``               ``bool``     True if rendered in memory
``context`` (v1.2, optional) ``(C, H, W)``        ``float32``  sidecar layers the architecture reads
``context_available``        ``(n_sidecars,)``    ``bool``     False -> that sidecar's channels are meaningless
``chip_id``                  —                    ``str``      manifest primary key
``basin_id``                 —                    ``str``      split key
============================ ==================== ============ ==============================

After default collation every tensor gains a leading batch axis, giving the
``(B, C, H, W)`` convention used throughout ``models/``.

Why ``optical`` is a tensor and never ``None``
----------------------------------------------
A ``None`` cannot be collated into a batch, and a batch mixing present and absent
optical is the normal case (``docs/02`` section 6). So the dataset always returns a
correctly shaped ``optical`` tensor and a separate boolean flag. **Where the flag is
False the tensor contents are meaningless** and the model substitutes the learned
missing-modality embedding (``docs/07`` section 7.2) rather than consuming them.

Two traps this module exists to make you confront
-------------------------------------------------
1. **Fork safety.** ``rasterio`` dataset handles are opened **inside the worker**, never
   in the parent process. A ``DataLoader`` with ``num_workers > 0`` forks; a handle opened
   before the fork is inherited with duplicated internal state and produces corrupted
   reads or a deadlock. Handles are cached in a bounded LRU keyed by ``(os.getpid(),
   path)``, so a forked worker misses the parent's cache and opens its own.
2. **Per-chip normalisation.** Never subtract a chip's own mean. A flooded chip is dark,
   and that is the signal. :func:`floodfusion.data.encoding.normalise` applies the frozen
   dataset-level statistics of :class:`~floodfusion.config.NormalizationStats`.

Where the pixels come from
--------------------------
``DataConfig.chip_source``: ``raster`` reads the 16-band event stacks (every path is
checked at construction, without opening a handle); ``synthetic`` renders each chip in
memory with :func:`floodfusion.data.synthetic.chip.synthesize_chip`; ``auto`` reads the
raster when the file exists and falls back to the renderer **only** when it does not —
never on a read error — with a :class:`SyntheticChipWarning` once per process and a
``synthetic`` flag on every sample, so simulated pixels can never pass unnoticed.

References
----------
``docs/05_dataset_construction_and_splits.md`` sections 4, 7, 8, 9.
``docs/04_data_access_and_sentinel1_preprocessing.md`` sections 2.3, 8.1.
``docs/08_training_recipe_and_vram.md`` section 6.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    DataConfig,
    GridConfig,
    NormalizationStats,
)
from floodfusion.data.encoding import N_CHIP_BANDS, decode_physical, normalise
from floodfusion.data.records import ChipRecord, load_manifest
from floodfusion.utils.shapes import assert_shape, named_dims

__all__ = [
    "BalancedChipSampler",
    "ChipRecord",
    "FloodChipDataset",
    "SyntheticChipWarning",
    "build_dataloader",
    "iter_batches_to_device",
    "load_manifest",
    "worker_init_fn",
]

LOGGER = logging.getLogger(__name__)

#: Upper bound on cached rasterio handles per process. Thousands of scene files with one
#: handle each would exhaust the file-descriptor limit; a few dozen is ample when chips are
#: windows of a handful of event stacks.
MAX_OPEN_HANDLES = 96   # the stack and up to two sidecars per event

#: Spatial tensors that every augmentation must transform identically.
_SPATIAL_KEYS = ("sar", "optical", "terrain", "label", "valid", "optical_quality", "landcover")
#: Synthetic clouds (``data.cloud_augment_p``): share of the chip covered, cloud-top reflectance,
#: and the smoothing scale of the random field (px).
CLOUD_COVER = (0.1, 0.7)
CLOUD_REFLECTANCE = (0.3, 0.8)
CLOUD_SMOOTH_PX = 16


class SyntheticChipWarning(RuntimeWarning):
    """Emitted (once per process) when a chip is rendered instead of read from disk."""


_WARNED_PIDS: set[int] = set()


class FloodChipDataset(Dataset[dict[str, Any]]):
    r"""Windowed-read dataset over co-registered flood chips.

    Parameters
    ----------
    records
        Chip records for **this split only**. Filtering by split happens in
        :mod:`floodfusion.splits`, not here — the dataset must not be able to see
        chips from another split, because that is how leakage gets in
        (``docs/05`` section 6).
    norm
        Frozen, training-split-derived normalisation statistics.
    data_cfg
        Chipping, sampling and pixel-source configuration.
    grid_cfg
        Grid constants, needed for the ``int16`` dB decoding and the nodata sentinel.
    training
        If ``True``, augmentation is applied. Augmentation is **never** applied at
        evaluation.
    context_px
        Extra pixels read on every side (boundless, nodata-filled at scene edges) so the
        evaluator can predict with context and score only the central chip — the
        overlap-and-crop edge handling of ``docs/05`` section 8.1. Zero for training.

    Notes
    -----
    **Augmentation policy** (``docs/05`` section 8.4). Flips and 90-degree rotations are
    applied — drawn once per sample and applied to every spatial tensor. Arbitrary
    rotation is not: SAR geometry has a look direction, so layover, shadow and the local
    incidence channel are azimuth-dependent. Brightness or contrast jitter on the dB
    channels is **forbidden**: absolute dB level is the signal.

    **Modality dropout is not applied here.** It is architecture, not augmentation
    (``docs/07`` section 7.1), and it lives in the training loop so the same dataset serves
    ablation rows 9, 10 and 11 unchanged.
    """

    def __init__(
        self,
        records: Sequence[ChipRecord],
        norm: NormalizationStats,
        data_cfg: DataConfig,
        grid_cfg: GridConfig,
        *,
        training: bool = False,
        context_px: int = 0,
        sidecars: Sequence[str] = (),
    ) -> None:
        from floodfusion.data.sidecars import SIDECARS

        unknown = [n for n in sidecars if n not in SIDECARS]
        if unknown:
            raise ValueError(f"unknown sidecar(s) {unknown}; known: {sorted(SIDECARS)}")
        self.sidecars = tuple(sidecars)
        self.records = tuple(records)
        self.norm = norm
        self.data_cfg = data_cfg
        self.grid_cfg = grid_cfg
        self.training = training
        self.context_px = int(context_px)

        # Per-process handle cache. Keyed by PID so that a handle opened in the parent
        # is never reused in a forked worker -- see the module docstring, trap 1.
        self._handles: dict[tuple[int, str], Any] = OrderedDict()

        if not self.records:
            raise ValueError("FloodChipDataset constructed with zero records")
        if self.context_px < 0:
            raise ValueError("context_px must be non-negative")
        if data_cfg.chip_source == "raster":
            missing = sorted({str(r.raster_path) for r in self.records if not Path(r.raster_path).is_file()})
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} raster(s) referenced by the manifest do not exist, e.g. "
                    f"{missing[:3]} (chip_source='raster' never substitutes simulated pixels)"
                )

    def __len__(self) -> int:
        return len(self.records)

    # -- handle management --------------------------------------------------
    def _get_handle(self, path: Path) -> Any:
        """Return a ``rasterio`` dataset handle valid **in the calling process**.

        The PID in the cache key is the whole point: after a ``fork`` the child inherits
        ``self._handles``, but its PID differs, so it misses the cache and opens its own
        handle (one ``open`` per worker per file). The cache is an LRU bounded by
        :data:`MAX_OPEN_HANDLES`.
        """
        import rasterio

        key = (os.getpid(), str(path))
        handle = self._handles.get(key)
        if handle is not None:
            self._handles.move_to_end(key)  # type: ignore[attr-defined]
            return handle
        handle = rasterio.open(path)
        self._handles[key] = handle
        while len(self._handles) > MAX_OPEN_HANDLES:
            _, old = self._handles.popitem(last=False)  # type: ignore[call-arg]
            if old is not None and not old.closed:
                old.close()
        return handle

    def __getstate__(self) -> dict[str, Any]:
        # Never pickle open handles into spawned workers.
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    # -- reading ------------------------------------------------------------
    def _use_synthetic(self, record: ChipRecord) -> bool:
        source = self.data_cfg.chip_source
        if source == "synthetic":
            return True
        if source == "raster":
            return False
        return not Path(record.raster_path).is_file()

    def _read_window(self, record: ChipRecord) -> np.ndarray:
        """Read the 16-band stack window for ``record`` as stored (``int16``).

        Returns
        -------
        numpy.ndarray
            ``(16, height + 2c, width + 2c)`` with ``c = context_px``. All bands are read
            in **one** call, through ``rasterio.windows.Window`` — never the whole scene
            and a slice (``docs/04`` section 2.3). The read is ``boundless`` with the
            nodata sentinel as fill, so context windows at a scene edge are full-sized and
            the padding is routed into the validity mask rather than silently shrinking
            the chip.
        """
        col, row, width, height = record.window
        c = self.context_px
        if self._use_synthetic(record):
            pid = os.getpid()
            if pid not in _WARNED_PIDS:
                _WARNED_PIDS.add(pid)
                warnings.warn(
                    f"rendering chips in memory (chip_source={self.data_cfg.chip_source!r}); "
                    f"e.g. {record.raster_path} is not read from disk",
                    SyntheticChipWarning, stacklevel=3,
                )
            from floodfusion.data.synthetic.chip import synthesize_chip

            raw = synthesize_chip(record, self.grid_cfg, self.norm, height=height + 2 * c, width=width + 2 * c)
        else:
            from rasterio.windows import Window

            src = self._get_handle(Path(record.raster_path))
            if src.count < N_CHIP_BANDS:
                raise ValueError(f"{record.raster_path}: expected {N_CHIP_BANDS} bands, found {src.count}")
            raw = src.read(
                indexes=list(range(1, N_CHIP_BANDS + 1)),
                window=Window(col - c, row - c, width + 2 * c, height + 2 * c),
                boundless=True,
                fill_value=self.grid_cfg.nodata_int16,
            )
        expected = (N_CHIP_BANDS, height + 2 * c, width + 2 * c)
        if raw.shape != expected:
            raise ValueError(f"{record.chip_id}: short read {raw.shape}, expected {expected}")
        return raw

    def _decode_and_normalise(self, raw: np.ndarray) -> dict[str, torch.Tensor]:
        r"""Decode ``int16`` storage and apply the frozen normalisation.

        The nodata sentinel is detected **before** any arithmetic and routed into the
        validity mask; Sentinel-2 DN are converted with
        :math:`\rho = (\mathrm{DN} + \texttt{BOA\_ADD\_OFFSET}) / \texttt{QUANTIFICATION\_VALUE}`;
        SAR is z-scored with the frozen per-channel statistics, reflectance clipped to
        [0, 1], MNDWI left in [-1, 1], terrain scaled by fixed physical constants; invalid
        positions are zero after normalisation. No statistic derived from ``raw`` itself
        is used anywhere (see :func:`floodfusion.data.encoding.normalise`).
        """
        physical = decode_physical(raw, self.grid_cfg, self.norm)
        arrays = normalise(physical, self.norm)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        out["landcover"] = torch.from_numpy(physical.landcover[None].copy())
        return out

    def _paste_cloud(self, sample: dict[str, Any]) -> dict[str, Any]:
        """A synthetic cloud on part of the optical chip (v1.1 row 9c): quality 0, bright reflectance.

        A smooth random field, thresholded to cover :data:`CLOUD_COVER` of the chip, marks the
        cloud; under it every optical band becomes a smooth bright value (SWIR a little darker,
        as for real cloud tops) and the quality drops to 0 — what the encoder meets on a real
        partly cloudy chip. The SAR, the label and the validity are untouched.
        """
        import torch.nn.functional as F

        quality = sample["optical_quality"]
        _, h, w = quality.shape
        cover = float(torch.empty(()).uniform_(*CLOUD_COVER))
        coarse = (max(2, h // CLOUD_SMOOTH_PX), max(2, w // CLOUD_SMOOTH_PX))
        field = F.interpolate(torch.randn(1, 1, *coarse), size=(h, w), mode="bilinear", align_corners=False)[0]
        cloud = field > torch.quantile(field.flatten(), 1.0 - cover)
        lo, hi = CLOUD_REFLECTANCE
        bright = F.interpolate(torch.rand(1, 1, *coarse), size=(h, w), mode="bilinear", align_corners=False)[0]
        bright = lo + (hi - lo) * bright
        cloud_bands = torch.cat([bright, bright, 0.8 * bright, torch.full_like(bright, 0.2 / 1.8)], dim=0)
        sample["optical"] = torch.where(cloud, cloud_bands.to(sample["optical"].dtype), sample["optical"])
        sample["optical_quality"] = torch.where(cloud, torch.zeros_like(quality), quality)
        return sample

    def _context(self, record: ChipRecord, sample: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The architecture's sidecar channels for this window (zeros where a sidecar is missing) and availability."""
        from floodfusion.data.sidecars import SIDECARS, normalise_sidecar, read_sidecar_window

        h, w = sample["sar"].shape[-2:]
        valid = sample["valid"][0].numpy()
        # The post-event VV in dB, recovered from the frozen normalisation (only the seasonal change needs it).
        vv_post = sample["sar"][0].numpy() * self.norm.sar_std_db[0] + self.norm.sar_mean_db[0]
        blocks, available = [], []
        for name in self.sidecars:
            spec = SIDECARS[name]
            path = Path(record.raster_path).with_name(spec.filename)
            values = None
            if not self._use_synthetic(record) and path.is_file():
                raw = read_sidecar_window(self._get_handle(path), record.window, self.context_px, spec.raw_bands)
                values = normalise_sidecar(name, raw, vv_post_db=np.where(valid, vv_post, np.nan), norm=self.norm)
            ok = values is not None and bool(np.isfinite(values).any())
            blocks.append(np.nan_to_num(values, nan=0.0) if ok else np.zeros((spec.n_channels, h, w), np.float32))
            available.append(ok)
        context = torch.from_numpy(np.ascontiguousarray(np.concatenate(blocks).astype(np.float32)))
        return context, torch.tensor(available)

    def _augment(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Flips and 90-degree rotations, drawn once, applied to every spatial tensor."""
        k = int(torch.randint(0, 4, ()).item())
        flip_h = bool(torch.randint(0, 2, ()).item())
        flip_v = bool(torch.randint(0, 2, ()).item())
        for key in (*_SPATIAL_KEYS, "context"):
            if key not in sample:
                continue
            t = sample[key]
            if k:
                t = torch.rot90(t, k, dims=(-2, -1))
            if flip_h:
                t = torch.flip(t, dims=(-1,))
            if flip_v:
                t = torch.flip(t, dims=(-2,))
            sample[key] = t.contiguous()
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample, obeying the contract in the module docstring."""
        record = self.records[index]
        raw = self._read_window(record)
        sample: dict[str, Any] = self._decode_and_normalise(raw)
        quality = sample["optical_quality"]
        available = bool(record.optical_available) and bool((quality > 0).any())
        if not available:
            sample["optical_quality"] = torch.zeros_like(quality)
        if self.training and available and self.data_cfg.cloud_augment_p > 0 and \
                float(torch.rand(())) < self.data_cfg.cloud_augment_p:
            sample = self._paste_cloud(sample)
        sample["optical_available"] = torch.tensor(available)
        sample["synthetic"] = torch.tensor(self._use_synthetic(record))
        if self.sidecars:
            sample["context"], sample["context_available"] = self._context(record, sample)
        sample["chip_id"] = record.chip_id
        sample["basin_id"] = record.basin_id
        if self.training:
            sample = self._augment(sample)
        dims = named_dims()
        assert_shape(sample["sar"], (N_SAR_CHANNELS, "H", "W"), dims, name="sar")
        assert_shape(sample["optical"], (N_OPTICAL_CHANNELS, "H", "W"), dims, name="optical")
        assert_shape(sample["terrain"], (N_TERRAIN_CHANNELS, "H", "W"), dims, name="terrain")
        for key in ("label", "valid", "optical_quality", "landcover"):
            assert_shape(sample[key], (1, "H", "W"), dims, name=key)
        if sample["valid"].dtype != torch.bool or sample["label"].dtype != torch.float32:
            raise TypeError("valid must be bool and label float32")
        return sample


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
class BalancedChipSampler(Sampler[int]):
    """Every positive chip each epoch, plus a fresh random draw of negatives.

    Most chips contain no flood at all (``docs/05`` section 8.2), so most batches would
    carry no positive gradient. Keeping every chip whose flood fraction exceeds
    ``positive_fraction_threshold`` and subsampling the rest to
    ``negative_to_positive_ratio`` per epoch fixes the chip-level imbalance — and **changes
    the prior**: :meth:`training_prior` reports the expected positive *pixel* fraction the
    model sees, which the evaluator uses for the prior correction of ``docs/05`` section 8.3
    before any threshold is applied. Deterministic for a given ``seed`` and epoch.
    """

    def __init__(
        self,
        records: Sequence[ChipRecord],
        *,
        negative_to_positive_ratio: float,
        positive_fraction_threshold: float,
        seed: int = 0,
    ) -> None:
        self.records = tuple(records)
        self.ratio = float(negative_to_positive_ratio)
        self.positive = [i for i, r in enumerate(self.records) if r.positive_fraction > positive_fraction_threshold]
        self.negative = [i for i, r in enumerate(self.records) if r.positive_fraction <= positive_fraction_threshold]
        if not self.positive:
            LOGGER.warning("BalancedChipSampler: no positive chips; sampling every chip uniformly")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def n_negative(self) -> int:
        if not self.positive:
            return len(self.negative)
        return min(len(self.negative), int(round(self.ratio * len(self.positive))))

    def __len__(self) -> int:
        return len(self.positive) + self.n_negative

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self.seed, self.epoch])
        negatives = rng.choice(self.negative, size=self.n_negative, replace=False).tolist() if self.n_negative else []
        indices = self.positive + negatives
        rng.shuffle(indices)
        return iter(int(i) for i in indices)

    def inclusion_probability(self) -> np.ndarray:
        """Per-record probability of appearing in an epoch."""
        p = np.zeros(len(self.records))
        p[self.positive] = 1.0
        if self.negative:
            p[self.negative] = self.n_negative / len(self.negative)
        return p

    def training_prior(self) -> float:
        """Expected flooded fraction of valid pixels under this sampling scheme."""
        p = self.inclusion_probability()
        valid = np.array([r.valid_fraction for r in self.records])
        pos = np.array([r.positive_fraction for r in self.records])
        denom = float((p * valid).sum())
        return float((p * valid * pos).sum() / denom) if denom > 0 else float("nan")


def natural_prior(records: Sequence[ChipRecord]) -> float:
    """Flooded fraction of valid pixels with no resampling (the deployment prior)."""
    valid = np.array([r.valid_fraction for r in records])
    pos = np.array([r.positive_fraction for r in records])
    return float((valid * pos).sum() / valid.sum()) if valid.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id: int) -> None:
    """Seed each worker distinctly and reproducibly.

    PyTorch already gives every worker a distinct ``torch.initial_seed()`` (base seed +
    worker id); deriving the ``numpy`` and ``random`` seeds from it means six workers no
    longer draw six identical "random" augmentation streams. Safe to call in the main
    process too (``get_worker_info()`` is then ``None``; the seed is simply the process
    seed).
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002 - seeds the GLOBAL legacy RNG third-party code may use
    random.seed(seed)


def build_dataloader(
    dataset: FloodChipDataset,
    data_cfg: DataConfig,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    sampler: Sampler[int] | None = None,
    seed: int | None = None,
) -> DataLoader[dict[str, Any]]:
    """Construct a ``DataLoader`` with the settings this project requires (``docs/08`` 6).

    ``persistent_workers`` and ``prefetch_factor`` are passed only when
    ``num_workers > 0`` (PyTorch rejects them otherwise); ``pin_memory`` only when CUDA
    exists (page-locked memory without a GPU is pure cost); ``drop_last`` keeps batch
    shapes fixed so ``cudnn.benchmark`` stays a win. A seeded generator makes the shuffle
    order reproducible.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if sampler is not None and shuffle:
        raise ValueError("pass either a sampler or shuffle=True, not both")
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=bool(data_cfg.pin_memory and torch.cuda.is_available()),
    )
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(int(seed))
    if data_cfg.num_workers > 0:
        kwargs["persistent_workers"] = data_cfg.persistent_workers
        kwargs["prefetch_factor"] = data_cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def iter_batches_to_device(
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    channels_last: bool = True,
) -> Iterator[Mapping[str, Any]]:
    """Yield batches moved to ``device``, 4-D float tensors in ``channels_last``.

    ``non_blocking=True`` overlaps the copy with compute only when the loader pins memory.
    Apply ``channels_last`` to **both** model and inputs or to neither (``docs/08`` 5.5).
    Strings (``chip_id``, ``basin_id``) pass through untouched.
    """
    for batch in loader:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                if channels_last and value.dim() == 4 and value.is_floating_point():
                    value = value.contiguous(memory_format=torch.channels_last)
            moved[key] = value
        yield moved
