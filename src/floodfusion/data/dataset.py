"""PyTorch ``Dataset`` and ``DataLoader`` for co-registered flood chips.

This is the module in which most of Project 01's silent bugs are possible, so read
``tests/test_dataset.py`` before writing a line: it is the specification.

The contract
------------
Every sample is a ``dict`` with fixed keys, dtypes and shapes. ``H = W = chip_size``.

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
False the tensor contents are meaningless** and the model must substitute the learned
missing-modality embedding (``docs/07`` section 7.2) rather than consume them.

Do not fill unavailable optical with zeros and rely on the model noticing: zero is a
legitimate normalised reflectance, so "dark" and "absent" would be indistinguishable —
and darkness is precisely the water signal.

Two traps that this module exists to make you confront
------------------------------------------------------
1. **Fork safety.** ``rasterio`` dataset handles must be opened **inside the worker**,
   never in the parent process. A ``DataLoader`` with ``num_workers > 0`` forks; a
   handle opened before the fork is inherited with duplicated internal state and
   produces corrupted reads or a deadlock — often only after twenty minutes and not
   deterministically. See ``ENVIRONMENT_SETUP.md`` section 7.2. The pattern below caches
   handles in a dict keyed by ``os.getpid()``.

2. **Per-chip normalisation.** Never subtract a chip's own mean. A flooded chip is dark,
   and that is the signal. Use the frozen dataset-level statistics in
   :class:`~floodfusion.config.NormalizationStats`. See ``docs/05`` section 7.1.

References
----------
``docs/05_dataset_construction_and_splits.md`` sections 4, 7, 8, 9.
``docs/04_data_access_and_sentinel1_preprocessing.md`` sections 2.3, 8.1.
``docs/08_training_recipe_and_vram.md`` section 6.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    DataConfig,
    GridConfig,
    NormalizationStats,
)

__all__ = [
    "ChipRecord",
    "FloodChipDataset",
    "build_dataloader",
    "load_manifest",
    "worker_init_fn",
]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChipRecord:
    """One row of the chip manifest.

    Immutable by design: a dataset whose records mutate at runtime cannot be
    reproduced. See ``docs/05`` section 9 for the full field list the manifest must
    record for ``P1-M6.3``.

    Attributes
    ----------
    chip_id
        Primary key. Stable across regenerations of the dataset.
    basin_id
        The split key. Splits partition **basins**, never chips
        (``docs/05`` section 6.4).
    event_id
        Flood event this chip belongs to.
    raster_path
        Path to the stacked 12-band chip raster, or to the parent scene if you read
        windows from full scenes rather than pre-cut chips.
    window
        ``(col_off, row_off, width, height)`` for a windowed read. Present so that the
        same manifest supports both a pre-cut-chip layout and a windowed-read layout;
        pick one and be consistent.
    positive_fraction
        Fraction of valid pixels labelled flood. Used for sampling and stratification.
    valid_fraction
        Fraction of pixels that are valid. Chips below
        ``DataConfig.min_valid_fraction`` are dropped.
    optical_available
        Whether a usable Sentinel-2 observation exists for this chip.
    relative_orbit, orbit_state
        Recorded so that ``tests/test_stac_query.py``'s pre/post matching invariant
        (``docs/04`` section 7) can be re-checked at dataset level.
    """

    chip_id: str
    basin_id: str
    event_id: str
    raster_path: Path
    window: tuple[int, int, int, int]
    positive_fraction: float
    valid_fraction: float
    optical_available: bool
    relative_orbit: int
    orbit_state: str


def load_manifest(path: Path | str) -> tuple[ChipRecord, ...]:
    """Read the chip manifest written by :mod:`floodfusion.data.chipping`.

    Parameters
    ----------
    path
        Path to the manifest (Parquet or CSV).

    Returns
    -------
    tuple[ChipRecord, ...]
        Immutable sequence of records, in a **deterministic** order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``chip_id`` values are not unique, or a required column is missing.

    Notes
    -----
    Sort by ``chip_id`` before returning. An order that depends on filesystem iteration
    is not reproducible across machines, and it will silently change which chips land
    in which batch — which changes your results in a way you cannot explain.

    TODO(student)
    -------------
    1. Read the manifest into a dataframe.
    2. Validate that every column of :class:`ChipRecord` is present, and raise naming
       **all** missing columns rather than the first.
    3. Assert ``chip_id`` uniqueness; a duplicate silently doubles a chip's weight.
    4. Sort by ``chip_id`` and construct the frozen records.
    """
    raise NotImplementedError(
        "Implement manifest loading. Required fields are listed in docs/05 section 9."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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
        Chipping and sampling configuration.
    grid_cfg
        Grid constants, needed for the ``int16`` dB decoding and the nodata sentinel.
    training
        If ``True``, augmentation is applied. Augmentation is **never** applied at
        evaluation.

    Notes
    -----
    **Augmentation policy** (``docs/05`` section 8.4). Flips and 90-degree rotations are
    safe. Arbitrary rotation is **not**: SAR geometry has a look direction, so layover,
    shadow and the local incidence angle channel are all azimuth-dependent, and rotating
    the raster without rotating that semantics manufactures physically impossible
    samples. Brightness or contrast jitter on the dB channels is **forbidden**: absolute
    dB level is the signal.

    **Modality dropout is not applied here.** It is architecture, not augmentation
    (``docs/07`` section 7.1), and it belongs in the training loop so that the same
    dataset can serve ablation rows 9, 10 and 11 unchanged.

    Examples
    --------
    The shape contract, expressed as the test asserts it::

        sample = dataset[0]
        assert sample["sar"].shape     == (5, 256, 256)
        assert sample["optical"].shape == (4, 256, 256)
        assert sample["label"].dtype   == torch.float32
        assert sample["valid"].dtype   == torch.bool
        assert torch.isfinite(sample["sar"]).all()
    """

    def __init__(
        self,
        records: Sequence[ChipRecord],
        norm: NormalizationStats,
        data_cfg: DataConfig,
        grid_cfg: GridConfig,
        *,
        training: bool = False,
    ) -> None:
        self.records = tuple(records)
        self.norm = norm
        self.data_cfg = data_cfg
        self.grid_cfg = grid_cfg
        self.training = training

        # Per-process handle cache. Keyed by PID so that a handle opened in the parent
        # is never reused in a forked worker -- see the module docstring, trap 1.
        self._handles: dict[tuple[int, str], Any] = {}

        if not self.records:
            raise ValueError("FloodChipDataset constructed with zero records")

    def __len__(self) -> int:
        return len(self.records)

    # -- handle management --------------------------------------------------
    def _get_handle(self, path: Path) -> Any:
        """Return a ``rasterio`` dataset handle valid **in the calling process**.

        Returns
        -------
        rasterio.DatasetReader
            An open handle, cached per ``(pid, path)``.

        Notes
        -----
        The PID in the cache key is the whole point. After a ``fork``, the child
        inherits ``self._handles`` containing handles whose GDAL-internal state belongs
        to the parent. Reusing one produces corrupted reads or a deadlock inside
        ``libhdf5``/GDAL, typically non-deterministically and after a long delay. Keying
        by PID means a forked worker misses the cache and opens its own handle, which is
        correct and costs one ``open`` per worker per file.

        TODO(student)
        -------------
        1. Build the key ``(os.getpid(), str(path))``.
        2. On a miss, ``rasterio.open(path)`` and store it.
        3. Return the handle.
        4. Consider bounding the cache: with thousands of scene files, one handle each
           will exhaust the file-descriptor limit. An LRU of a few dozen is ample.
        """
        raise NotImplementedError(
            "Implement per-process handle caching. See ENVIRONMENT_SETUP.md section 7.2 "
            "for the fork-safety failure this prevents."
        )

    # -- reading ------------------------------------------------------------
    def _read_window(self, record: ChipRecord) -> Any:
        """Read the 12-band chip for ``record`` as a raw array.

        Returns
        -------
        numpy.ndarray
            Shape ``(12, H, W)``, dtype ``int16`` as stored on disk.

        Notes
        -----
        Use ``rasterio.windows.Window(col_off, row_off, width, height)`` and pass it as
        the ``window=`` argument. Do **not** read the whole scene and slice: that
        transfers the entire raster and defeats the internal tiling that makes a
        windowed read cheap (``docs/04`` section 2.3).

        Align read windows to the file's internal tile grid where you can. A window
        straddling four internal tiles fetches four tiles.

        TODO(student)
        -------------
        1. Construct the ``Window`` from ``record.window``.
        2. Read all 12 bands in one call — one call reading 12 bands is far cheaper
           than 12 calls reading one band.
        3. Assert the returned shape is ``(12, height, width)``; a short read at a
           scene edge returns a smaller array **without raising**, and that silently
           produces a mis-sized chip. Use ``boundless=True`` with an explicit fill, or
           reject the record.
        """
        raise NotImplementedError(
            "Implement the windowed read. See docs/04 section 2.3 for why windows and "
            "not slices."
        )

    def _decode_and_normalise(self, raw: Any) -> dict[str, torch.Tensor]:
        r"""Decode ``int16`` storage and apply frozen normalisation.

        Parameters
        ----------
        raw
            ``(12, H, W)`` ``int16`` array as stored.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys ``sar``, ``optical``, ``terrain``, each ``float32``.

        Notes
        -----
        **Decoding.** dB channels are stored as :math:`\mathrm{round}(100 \cdot
        \gamma^0_{\mathrm{dB}})` (``docs/04`` section 8.1), so divide by
        ``grid_cfg.db_scale``. The nodata sentinel ``grid_cfg.nodata_int16`` must be
        detected **before** the division and routed into the validity mask — a sentinel
        that survives into the float array becomes :math:`-327.68` dB, which then
        normalises to an enormous value and trains the model to detect scene borders.

        **Sentinel-2 decoding** (``docs/05`` section 7.2)::

            rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE

        Omitting ``BOA_ADD_OFFSET`` shifts every reflectance by 0.1, which is larger
        than the water reflectances of 0.02--0.06 you are trying to measure.

        **Normalisation** (``docs/05`` section 7.2), per modality, never per chip:

        - SAR dB: z-score with the frozen per-channel ``sar_mean_db`` / ``sar_std_db``.
        - Optical: fixed physical scaling, then clip to :math:`[0, 1]`.
        - MNDWI: already in :math:`[-1, 1]`; leave it alone.
        - Terrain: fixed physical scaling by ``slope_scale_deg``, ``hand_log_scale``,
          ``incidence_scale_deg``.

        Fill invalid positions with **zero after normalisation**, so they sit at the
        dataset mean and contribute nothing once masked out of the loss.

        TODO(student)
        -------------
        1. Build the nodata mask from ``raw`` before any arithmetic.
        2. Decode each modality group per the rules above.
        3. Apply the frozen statistics. Assert you have not used any statistic derived
           from ``raw`` itself.
        4. Convert to ``float32`` ``torch.Tensor`` and return the three groups.
        5. Assert no ``NaN``/``Inf`` survives — ``utils.shapes.check_finite`` while
           debugging.
        """
        raise NotImplementedError(
            "Implement decoding and normalisation. See docs/05 section 7 — and note "
            "that per-chip standardisation is forbidden there."
        )

    def _augment(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply geometric augmentation, training only.

        Notes
        -----
        Permitted: horizontal flip, vertical flip, 90-degree rotations. These must be
        applied **identically** to every spatial tensor in the sample — ``sar``,
        ``optical``, ``terrain``, ``label``, ``valid``, ``optical_quality``. Augmenting
        the image and forgetting the label is a bug that produces a model that trains to
        a plateau and never improves, with no error message.

        Forbidden, and why: arbitrary rotation (breaks SAR azimuth semantics), any
        radiometric jitter on dB channels (destroys the absolute level that is the
        signal). See ``docs/05`` section 8.4.

        TODO(student)
        -------------
        1. Draw the transform once per sample.
        2. Apply it to every spatial tensor in one place, so that adding a new tensor
           to the sample cannot forget to augment it.
        3. Add a test asserting that ``label`` and ``sar`` receive the same transform —
           the cheapest insurance against the bug above.
        """
        raise NotImplementedError("Implement augmentation. See docs/05 section 8.4.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample, obeying the contract in the module docstring.

        TODO(student)
        -------------
        1. ``record = self.records[index]``.
        2. ``raw = self._read_window(record)``.
        3. ``groups = self._decode_and_normalise(raw)``.
        4. Build ``label`` ``(1, H, W)`` ``float32`` and ``valid`` ``(1, H, W)`` ``bool``
           from the validity flags — valid iff the flag byte is exactly
           :attr:`~floodfusion.config.ValidityFlag.VALID`.
        5. Build ``optical_quality`` and the scalar ``optical_available`` from
           ``record.optical_available`` and the cloud/shadow mask.
        6. If ``self.training``, augment.
        7. Assert every shape and dtype with ``utils.shapes.assert_shape`` before
           returning. This assertion is cheap and has caught more bugs in this project
           than any other single line.
        """
        raise NotImplementedError(
            "Implement __getitem__. The contract is the table in this module's "
            "docstring, and tests/test_dataset.py asserts it."
        )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id: int) -> None:
    """Seed each worker distinctly and reproducibly.

    Notes
    -----
    Without this, every worker inherits the parent's RNG state and draws the **same**
    augmentation sequence — so with ``num_workers=6`` you get six identical streams of
    "random" flips. The effect is a quiet reduction in augmentation diversity that never
    raises and is very hard to notice.

    TODO(student)
    -------------
    1. Derive a per-worker seed from ``torch.initial_seed()`` (which PyTorch already
       makes worker-distinct) modulo :math:`2^{32}`.
    2. Seed ``numpy`` and the ``random`` module with it.
    """
    raise NotImplementedError("Implement worker seeding. See docs/08 section 7.2.")


def build_dataloader(
    dataset: FloodChipDataset,
    data_cfg: DataConfig,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader[dict[str, Any]]:
    """Construct a ``DataLoader`` with the settings this project requires.

    Notes
    -----
    The settings and their reasons (``docs/08`` section 6):

    ``num_workers``
        4--8. On this machine the dataloader is frequently the bottleneck, not the GPU.
        Measure; do not guess.
    ``pin_memory=True``
        Page-locked staging buffers enable asynchronous host-to-device DMA. Only
        effective if the transfer also passes ``non_blocking=True``. Costs
        ``num_workers * prefetch_factor * batch_bytes`` of unswappable host RAM — around
        2.4 GB at the reference settings, which is a reason to store chips as ``int16``.
    ``persistent_workers=True``
        Avoids re-forking every epoch, which is both slow and a repeated opportunity for
        the fork-safety trap.
    ``drop_last=True``
        Keeps batch shapes fixed, which is what makes ``cudnn.benchmark = True`` a win
        rather than a repeated retuning cost.

    TODO(student)
    -------------
    1. Construct the ``DataLoader`` with the above and ``worker_init_fn``.
    2. Confirm ``persistent_workers`` is only set when ``num_workers > 0``; PyTorch
       raises otherwise.
    3. Time one epoch against a synthetic in-memory dataset to determine whether you
       are input-bound (``docs/08`` section 6).
    """
    raise NotImplementedError("Implement the DataLoader. See docs/08 section 6.")


def iter_batches_to_device(
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    channels_last: bool = True,
) -> Iterator[Mapping[str, Any]]:
    """Yield batches moved to ``device``, optionally in ``channels_last`` layout.

    Notes
    -----
    Use ``.to(device, non_blocking=True)``, which only overlaps with compute when the
    loader was built with ``pin_memory=True``.

    Apply ``channels_last`` to **both** model and inputs or to neither. Half-applied, it
    inserts layout conversions on every forward and is slower than not applying it at
    all (``docs/08`` section 5.5).

    Move only tensors: ``chip_id`` and ``basin_id`` are strings and must be passed
    through untouched.

    TODO(student)
    -------------
    1. Iterate the loader.
    2. Move tensor values; pass non-tensors through.
    3. Convert 4-D float tensors to ``torch.channels_last`` when requested.
    """
    raise NotImplementedError("Implement device transfer. See docs/08 sections 5.5, 6.")
