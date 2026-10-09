r"""Chip extraction and manifest generation.

Milestone ``P1-M1``, final step. Produces the chips and the immutable manifest that
:mod:`floodfusion.data.dataset` consumes and :mod:`floodfusion.splits` partitions.

References
----------
``docs/05_dataset_construction_and_splits.md`` sections 8, 9.
``docs/03_dem_hand_and_terrain_effects.md`` sections 3, 7.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["build_analysis_grid", "extract_chips", "write_manifest"]


def build_analysis_grid(bbox: tuple[float, float, float, float], pixel_size_m: float = 10.0):
    r"""Define the single analysis grid for one event.

    Notes
    -----
    Every layer -- SAR, optical, DEM derivatives, labels -- must end on **one grid**. Not
    "the same projection": one grid, so that array index :math:`(i,j)` denotes the same
    patch of ground in every layer.

    - **CRS**: the UTM zone containing the AOI centroid. Metric and conformal, which the
      Horn slope of ``docs/03`` section 4 requires. An AOI spanning a zone boundary needs
      a *decision*, not a silent default -- pick one zone and accept the distortion, or
      split the AOI, and record which.
    - **Pixel size**: 10 m, native to S1 GRD-HD and S2 B3/B8.
    - **Origin**: snapped to a multiple of the pixel size,
      :math:`c = \lfloor x_{\min}/10 \rfloor \times 10`,
      :math:`f = \lceil y_{\max}/10 \rceil \times 10`. Snapping makes the grid a
      deterministic function of the AOI alone, so two layers processed independently --
      months apart, by different people -- land on exactly the same grid with no
      resampling phase difference. Without it, two grids can be offset by an arbitrary
      fraction of a pixel, and every later resampling smears that offset into the data
      where it can never be recovered.

    .. warning::

       **The half-pixel convention.** In GDAL and ``rasterio``, the transform origin is
       the **upper-left corner of the upper-left pixel**, not its centre; the centre of
       pixel (0,0) is at :math:`(c + a/2,\ f + e/2)`. Some formats and many hand-rolled
       scripts use the centre convention. Mixing them offsets everything by **5 m on a
       10 m grid** -- half of ``P1-M1``'s entire one-pixel error budget, spent for free
       and in a systematic direction.

       Also note that ``rasterio`` orders the affine as ``Affine(a, b, c, d, e, f)``
       while the GDAL geotransform tuple is ``(c, a, b, f, d, e)``. They are **not** the
       same order, and transposing them silently places the raster in the wrong
       hemisphere.

    TODO(student)
    -------------
    1. Derive the UTM zone from the centroid; construct the CRS.
    2. Snap the origin; expand the extent to whole pixels.
    3. Return CRS, transform, width and height.
    4. Add the ``P1-M1`` check: read a recognisable point feature -- a bridge, a dam, a
       small island -- from the SAR and optical grids independently. They must agree to
       well within 10 m. A disagreement of almost exactly 5 m is a convention error, not
       a co-registration error, and no resampling will fix it.
    """
    raise NotImplementedError("Implement the analysis grid. See docs/03 section 3.")


def extract_chips(*args: object, **kwargs: object) -> None:
    r"""Cut the co-registered stack into chips and write them.

    Notes
    -----
    **Resampling rules** (``docs/03`` section 7), determined by the *type* of the
    quantity and not by convenience:

    ========================== ==================== =====================================
    layer type                 kernel               why
    ========================== ==================== =====================================
    continuous physical        **bilinear**         smooth, no overshoot
    categorical (SCL, basins)  **nearest, always**  see the warning
    binary masks               **nearest**          must stay in {0,1}
    downsampling > 2x          **average / area**   anti-aliasing
    ========================== ==================== =====================================

    Cubic and Lanczos are excluded deliberately: both overshoot at discontinuities. At a
    land--water edge in a dB image, cubic undershoots produce values below the true
    minimum that look exactly like darker water and widen the apparent water body.

    .. warning::

       **Never bilinearly interpolate a class index.** Interpolating between SCL class 3
       (cloud shadow) and class 9 (cloud high probability) yields :math:`(3+9)/2 = 6`,
       which is **WATER**. The result is a thin ring of spurious water around every cloud
       in every scene -- systematically, at exactly the boundaries where your validity
       mask is already uncertain. ``docs/02`` exercise 6 asks you to construct this bug
       and observe it.

    **Resample once.** Each resampling is a low-pass filter; applying it repeatedly
    smooths the data and accumulates sub-pixel positioning error. Establish the target
    grid first, then bring every source onto it in a **single** operation -- do not
    reproject the DEM, resample to 10 m, compute slope, then resample again to snap.

    **Chip geometry** (``docs/05`` section 8.1): 256x256 (2.56 km); stride 128 for
    training, 256 for evaluation so every pixel is scored exactly once. Overlapping
    training chips inflate apparent dataset size without adding independent information
    -- they are a mild augmentation, not new evidence, and must not be counted as
    independent samples in any statistical statement.

    **Chip before or after splitting?** Split by **basin first, then chip within each
    basin**. Chipping first with overlap and splitting after lets two overlapping chips
    land on opposite sides of a split boundary, sharing literal pixels
    (``docs/05`` section 6.5 point 3). That is debugging challenge C1's most likely root
    cause.

    TODO(student)
    -------------
    1. Bring every layer onto the analysis grid with the correct kernel, in one pass.
    2. Cut chips at the configured stride.
    3. Store as ``int16`` with dB x 100 and an explicit nodata sentinel outside the
       physical range (``docs/04`` section 8.1) -- this halves the disk footprint against
       ``float32`` at 0.01 dB quantisation, far finer than the 2.17 dB speckle noise.
    4. Record ``positive_fraction`` and ``valid_fraction`` per chip; drop chips below
       ``DataConfig.min_valid_fraction``.
    """
    raise NotImplementedError("Implement chipping. See docs/05 section 8.1.")


def write_manifest(records: object, path: Path | str) -> None:
    """Write the immutable chip manifest.

    Notes
    -----
    Fields required for ``P1-M6.3`` reproducibility (``docs/05`` section 9): ``chip_id``,
    ``event_id``, ``basin_id``, source scene IDs (S1 pre, S1 post, S2, DEM tiles),
    ``relative_orbit``, ``orbit_state``, grid CRS/transform/bounds, ``positive_fraction``,
    ``valid_fraction``, ``optical_available``, AOI cloud fraction, processing config hash
    and git SHA.

    **Write once; treat it as immutable.** If you must regenerate the dataset, bump a
    version string and re-run everything -- do not patch chips in place. ``P1-M6``'s
    reproducibility check regenerates the ablation table from a clean clone, and a
    mutated dataset defeats it silently.

    TODO(student)
    -------------
    1. Write Parquet (or CSV) sorted by ``chip_id``.
    2. Write the manifest's own content hash alongside, so
       :func:`floodfusion.splits.freeze_splits` can bind a split to a specific dataset.
    3. Write atomically -- temporary file then rename.
    """
    raise NotImplementedError("Implement the manifest. See docs/05 section 9.")
