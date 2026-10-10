r"""The analysis grid, resampling onto it, the event stack, and chip extraction.

Milestone ``P1-M1``, final step. Produces the event stacks, the chips (windows of them)
and the immutable manifest that :mod:`floodfusion.data.dataset` consumes and
:mod:`floodfusion.splits` partitions.

Torch-free; ``rasterio``, ``pyproj`` and ``shapely`` are imported inside the functions
that need them, so the arithmetic tests run on a bare machine.

References
----------
``docs/05_dataset_construction_and_splits.md`` sections 8, 9.
``docs/03_dem_hand_and_terrain_effects.md`` sections 3, 7.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import DataConfig, GridConfig
from floodfusion.data.encoding import N_CHIP_BANDS, QUALITY_SCALE, ChipBand
from floodfusion.data.records import ChipRecord, write_manifest

__all__ = [
    "MIN_OPTICAL_CLEAR_FRACTION",
    "AnalysisGrid",
    "build_analysis_grid",
    "enumerate_windows",
    "extract_chips",
    "read_event_stack",
    "reproject_to_grid",
    "utm_epsg_for",
    "write_event_stack",
    "write_manifest",
]

LOGGER = logging.getLogger(__name__)

#: A chip counts as having usable optical when at least this fraction of it is clear.
#: Below that the per-pixel gate would see almost nothing but cloud; treat as absent.
MIN_OPTICAL_CLEAR_FRACTION: float = 0.10
#: Optical quality >= 0.5 counts as "clear" for the per-chip clear fraction.
CLEAR_QUALITY: float = 0.5


@dataclass(frozen=True)
class AnalysisGrid:
    """One grid for every layer of one event: CRS, north-up affine, size.

    ``transform`` is the rasterio/affine order ``(a, b, c, d, e, f)`` — **not** the GDAL
    geotransform ``(c, a, b, f, d, e)``. ``(c, f)`` is the upper-left **corner** of the
    upper-left pixel; pixel ``(0, 0)``'s centre is ``(c + a/2, f + e/2)``.
    """

    epsg: int
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    notes: tuple[str, ...] = field(default=(), compare=False)

    @property
    def pixel_size(self) -> float:
        return float(self.transform[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` in the grid CRS."""
        a, _, c, _, e, f = self.transform
        return (c, f + e * self.height, c + a * self.width, f)

    @property
    def affine(self) -> Any:
        from affine import Affine

        return Affine(*self.transform)

    def pixel_centre(self, row: float, col: float) -> tuple[float, float]:
        a, _, c, _, e, f = self.transform
        return (c + (col + 0.5) * a, f + (row + 0.5) * e)


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing ``(lon, lat)`` (326xx north, 327xx south).

    The regular 6-degree zones; the Norway/Svalbard exceptions are deliberately ignored —
    the grid only needs to be metric and conformal, not the national convention.
    """
    if not -180.0 <= lon <= 180.0 or not -80.0 <= lat <= 84.0:
        raise ValueError(f"({lon}, {lat}) is outside the UTM domain")
    zone = min(int(math.floor((lon + 180.0) / 6.0)) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def build_analysis_grid(
    bbox: tuple[float, float, float, float],
    pixel_size_m: float = 10.0,
    *,
    epsg: int | None = None,
    snap_m: float | None = None,
) -> AnalysisGrid:
    r"""Define the single analysis grid for one event.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    pixel_size_m
        10 m, native to S1 GRD-HD and S2 B3/B8.
    epsg
        Force a projected CRS; ``None`` derives the UTM zone of the AOI centroid.
    snap_m
        Origin snapping step; defaults to the pixel size.

    Notes
    -----
    Every layer -- SAR, optical, DEM derivatives, labels -- must end on **one grid**. Not
    "the same projection": one grid, so that array index :math:`(i,j)` denotes the same
    patch of ground in every layer.

    - **CRS**: the UTM zone containing the AOI centroid. Metric and conformal, which the
      Horn slope of ``docs/03`` section 4 requires. An AOI spanning a zone boundary needs
      a *decision*, not a silent default: this function uses the centroid's zone and
      records the fact in ``AnalysisGrid.notes`` (and the log).
    - **Pixel size**: 10 m.
    - **Origin**: snapped to a multiple of the pixel size,
      :math:`c = \lfloor x_{\min}/10 \rfloor \times 10`,
      :math:`f = \lceil y_{\max}/10 \rceil \times 10`, and the extent expanded to whole
      pixels. Snapping makes the grid a deterministic function of the AOI alone, so two
      layers processed independently land on exactly the same grid.

    .. warning::

       **The half-pixel convention.** In GDAL and ``rasterio`` the transform origin is the
       **upper-left corner of the upper-left pixel**, not its centre. Mixing conventions
       offsets everything by **5 m on a 10 m grid** — half of ``P1-M1``'s one-pixel error
       budget. ``rasterio`` orders the affine ``Affine(a, b, c, d, e, f)``; the GDAL
       geotransform is ``(c, a, b, f, d, e)``.
    """
    from pyproj import CRS, Transformer

    west, south, east, north = (float(v) for v in bbox)
    if not (west < east and south < north):
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox}")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    snap = float(snap_m if snap_m is not None else pixel_size_m)
    notes: list[str] = []
    lon_c, lat_c = 0.5 * (west + east), 0.5 * (south + north)
    if epsg is None:
        epsg = utm_epsg_for(lon_c, lat_c)
        if utm_epsg_for(west, lat_c) != utm_epsg_for(east, lat_c):
            notes.append(
                f"AOI spans a UTM zone boundary; using the centroid zone EPSG:{epsg} and "
                "accepting the distortion at the far edge (docs/03 section 3.1)"
            )
            LOGGER.warning(notes[-1])
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    xmin, ymin, xmax, ymax = transformer.transform_bounds(west, south, east, north, densify_pts=21)
    c = math.floor(xmin / snap) * snap
    f = math.ceil(ymax / snap) * snap
    x_end = math.ceil(xmax / snap) * snap
    y_end = math.floor(ymin / snap) * snap
    width = int(math.ceil((x_end - c) / pixel_size_m - 1e-9))
    height = int(math.ceil((f - y_end) / pixel_size_m - 1e-9))
    return AnalysisGrid(
        epsg=int(epsg),
        transform=(float(pixel_size_m), 0.0, float(c), 0.0, -float(pixel_size_m), float(f)),
        width=width,
        height=height,
        notes=tuple(notes),
    )


def enumerate_windows(height: int, width: int, chip_size: int, stride: int) -> list[tuple[int, int]]:
    """``(row_off, col_off)`` of every **complete** window at the given stride.

    Partial windows at the scene edge are not produced: a short read returns a smaller
    array without raising and would silently produce a mis-sized chip.
    """
    if chip_size < 1 or stride < 1:
        raise ValueError("chip_size and stride must be positive")
    rows = range(0, height - chip_size + 1, stride)
    cols = range(0, width - chip_size + 1, stride)
    return [(r, c) for r in rows for c in cols]


_RESAMPLING_BY_KIND = {
    # docs/03 section 7.1 — determined by the TYPE of the quantity, never by convenience.
    "continuous": "bilinear",
    "categorical": "nearest",
    "binary": "nearest",
    "downsample": "average",
}


def reproject_to_grid(
    source: Any,
    grid: AnalysisGrid,
    *,
    kind: str,
    band: int | Sequence[int] = 1,
    nodata: float | None = None,
    src_nodata: float | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Bring one source raster onto the analysis grid in a **single** resampling.

    Parameters
    ----------
    source
        A path/URL readable by rasterio (``/vsicurl/`` COGs included) or an open dataset.
    kind
        ``continuous`` (bilinear), ``categorical`` / ``binary`` (nearest — never interpolate
        a class index: SCL 3 and 9 average to 6 = WATER, ``docs/02`` section 8), or
        ``downsample`` (average; anti-aliasing when reducing by > 2x). Cubic and Lanczos
        are excluded deliberately: they overshoot at land--water edges.

    Notes
    -----
    Uses a ``WarpedVRT`` onto the target grid, so only the source pixels needed for the
    grid are read, and the source is resampled exactly once. SAR must be passed in
    **linear power** (``docs/04`` section 3.4): bilinear weights averaged in dB are
    biased low.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    if kind not in _RESAMPLING_BY_KIND:
        raise ValueError(f"kind must be one of {sorted(_RESAMPLING_BY_KIND)}, got {kind!r}")
    resampling = getattr(Resampling, _RESAMPLING_BY_KIND[kind])
    opened = isinstance(source, str | Path)
    # Every source this pipeline reads is a GeoTIFF/COG: pinning the driver stops a file named
    # ".tif" that really holds VRT XML from making GDAL open other files or URLs.
    src = rasterio.open(source, driver="GTiff") if opened else source
    try:
        vrt_kwargs: dict[str, Any] = dict(
            crs=CRS.from_epsg(grid.epsg), transform=grid.affine,
            width=grid.width, height=grid.height, resampling=resampling,
        )
        if src_nodata is not None:
            vrt_kwargs["src_nodata"] = src_nodata
        if nodata is not None:
            vrt_kwargs["nodata"] = nodata
        with WarpedVRT(src, **vrt_kwargs) as vrt:
            bands = [band] if isinstance(band, int) else list(band)
            data = vrt.read(bands, out_dtype=dtype, masked=True)
    finally:
        if opened:
            src.close()
    fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else (nodata if nodata is not None else 0)
    out = data.filled(fill)
    return out[0] if isinstance(band, int) else out


_BAND_DESCRIPTIONS = (
    "gamma0_VV_post_dBx100", "gamma0_VH_post_dBx100", "gamma0_VV_pre_dBx100",
    "gamma0_VH_pre_dBx100", "delta_VV_dBx100", "S2_B3_DN", "S2_B8_DN", "S2_B11_DN",
    "MNDWIx1e4", "slope_degx100", "log1p_HANDx1e3", "local_incidence_degx100",
    "label", "validity_flags", "optical_qualityx1e4", "worldcover_class",
)


def write_event_stack(
    path: Path | str,
    stack: np.ndarray,
    grid: AnalysisGrid,
    *,
    nodata: int = -32768,
    tags: Mapping[str, Any] | None = None,
    block_size: int = 128,
) -> Path:
    """Write the 16-band ``int16`` event stack as a tiled, DEFLATE-compressed GeoTIFF.

    Internal tiles of 128 px align with both the evaluation stride (256) and the training
    stride (128), so every chip read touches whole tiles only (``docs/04`` section 2.3).
    Written to a temporary name and renamed, so a crash cannot leave a half-written stack.
    """
    import os

    import rasterio
    from rasterio.crs import CRS

    path = Path(path)
    stack = np.asarray(stack)
    if stack.dtype != np.int16 or stack.shape != (N_CHIP_BANDS, grid.height, grid.width):
        raise ValueError(
            f"stack must be int16 ({N_CHIP_BANDS}, {grid.height}, {grid.width}), "
            f"got {stack.dtype} {stack.shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    profile = dict(
        driver="GTiff", dtype="int16", count=N_CHIP_BANDS, width=grid.width, height=grid.height,
        crs=CRS.from_epsg(grid.epsg), transform=grid.affine, nodata=nodata, tiled=True,
        blockxsize=block_size, blockysize=block_size, compress="deflate", predictor=2,
        BIGTIFF="IF_SAFER", interleave="pixel",
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(_BAND_DESCRIPTIONS, start=1):
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, path)
    return path


def read_event_stack(path: Path | str) -> tuple[np.ndarray, AnalysisGrid, dict[str, str]]:
    """Read a whole event stack: ``(stack int16 (16,H,W), grid, tags)``."""
    import rasterio

    with rasterio.open(path) as src:
        stack = src.read()
        grid = AnalysisGrid(
            epsg=int(src.crs.to_epsg()), transform=tuple(src.transform)[:6],
            width=src.width, height=src.height,
        )
        tags = src.tags()
    return stack, grid, tags


def _integral(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    out[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return out


def _box_sum(integral: np.ndarray, r: int, c: int, size: int) -> int:
    return int(integral[r + size, c + size] - integral[r, c + size] - integral[r + size, c] + integral[r, c])


def extract_chips(
    stack_path: Path | str,
    *,
    event_id: str,
    basins: Mapping[str, Any],
    data_cfg: DataConfig,
    grid_cfg: GridConfig,
    relative_orbit: int,
    orbit_state: str,
    has_optical: bool,
) -> tuple[list[ChipRecord], dict[str, int]]:
    r"""Enumerate chips as windows of an event stack and record their statistics.

    Parameters
    ----------
    basins
        ``basin_id -> shapely geometry`` in the stack's CRS. A chip belongs to the basin
        containing its **centroid** (``docs/05`` section 6.4 step 2); chips whose centroid
        lies in no basin are dropped and counted.

    Returns
    -------
    records, counts
        One :class:`ChipRecord` per window at ``data_cfg.train_stride`` that passes
        ``min_valid_fraction``; ``eval_grid`` marks the non-overlapping windows at
        ``eval_stride``. ``counts`` reports windows considered and dropped per reason.

    Notes
    -----
    **Chip geometry** (``docs/05`` section 8.1): 256x256 (2.56 km); stride 128 for
    training, 256 for evaluation so every pixel is scored exactly once. Overlapping
    training chips are a mild augmentation, not new evidence, and are never counted as
    independent samples in any statistical statement.

    **Split by basin first, then chip** (``docs/05`` section 6.5 point 3): the windows are
    enumerated here, but which windows a split may use is decided by
    :func:`floodfusion.splits.assign_basins`, which drops every window whose footprint
    crosses a basin boundary — so no two windows on opposite sides of a split share a
    pixel.

    Per-window fractions come from integral images (O(1) per window), so enumerating a
    full scene at stride 128 costs one pass over the label, validity and quality bands.
    """
    import rasterio
    from shapely.geometry import Point
    from shapely.prepared import prep

    chip, stride, eval_stride = data_cfg.chip_size, data_cfg.train_stride, data_cfg.eval_stride
    if eval_stride % stride != 0:
        raise ValueError("eval_stride must be a multiple of train_stride so eval windows are a subset")
    nodata = int(grid_cfg.nodata_int16)
    with rasterio.open(stack_path) as src:
        bands = src.read([int(ChipBand.VV_POST) + 1, int(ChipBand.LABEL) + 1,
                          int(ChipBand.VALIDITY) + 1, int(ChipBand.OPTICAL_QUALITY) + 1])
        transform = tuple(src.transform)[:6]
        height, width = src.height, src.width
    vv, label, validity, quality = bands
    valid = (validity == 0) & (label != nodata) & (vv != nodata)
    positive = valid & (label == 1)
    clear = quality >= int(round(CLEAR_QUALITY * QUALITY_SCALE))
    i_valid, i_pos, i_clear = _integral(valid), _integral(positive), _integral(clear)
    prepared = {bid: prep(geom) for bid, geom in basins.items()}
    a, _, c0, _, e, f0 = transform
    area = chip * chip
    records: list[ChipRecord] = []
    counts = {"windows": 0, "dropped_low_valid": 0, "dropped_no_basin": 0}
    for row, col in enumerate_windows(height, width, chip, stride):
        counts["windows"] += 1
        n_valid = _box_sum(i_valid, row, col, chip)
        valid_fraction = n_valid / area
        if valid_fraction < data_cfg.min_valid_fraction:
            counts["dropped_low_valid"] += 1
            continue
        cx = c0 + (col + chip / 2.0) * a
        cy = f0 + (row + chip / 2.0) * e
        centre = Point(cx, cy)
        basin_id = next((bid for bid, geom in prepared.items() if geom.contains(centre)), None)
        if basin_id is None:
            counts["dropped_no_basin"] += 1
            continue
        clear_fraction = _box_sum(i_clear, row, col, chip) / area
        records.append(ChipRecord(
            chip_id=f"{event_id}_r{row:05d}_c{col:05d}",
            basin_id=str(basin_id),
            event_id=event_id,
            raster_path=Path(stack_path),
            window=(col, row, chip, chip),
            positive_fraction=_box_sum(i_pos, row, col, chip) / max(n_valid, 1),
            valid_fraction=valid_fraction,
            optical_available=bool(has_optical and clear_fraction >= MIN_OPTICAL_CLEAR_FRACTION),
            relative_orbit=int(relative_orbit),
            orbit_state=str(orbit_state),
            eval_grid=(row % eval_stride == 0 and col % eval_stride == 0),
            optical_clear_fraction=float(clear_fraction),
            centroid_x=float(cx),
            centroid_y=float(cy),
        ))
    LOGGER.info("%s: %d chips from %d windows (%s)", event_id, len(records), counts["windows"], counts)
    return records, counts
