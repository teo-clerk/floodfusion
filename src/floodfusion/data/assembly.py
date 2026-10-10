r"""Assemble one analysis-ready stack on one grid from matched archive items.

Shared by the two real-data routes:

- :mod:`floodfusion.data.ingest` — one flood event, labelled by a CEMS (or any) vector;
- :mod:`floodfusion.data.sen1floods11` — the Sen1Floods11 hand-labelled chips, whose
  labels are kept and whose inputs are re-fetched through this same path.

What a route supplies is a :class:`GridTask`: the grid, the matched Sentinel-1 pair, the
acquisition geometry, the reference layers (label, AOI, permanent water, land cover) and
the split groups. Everything else — Route B gamma0, terrain, optical selection, the
validity band, the int16 stack, chips, manifest and the per-grid QA — is identical.

**Terrain margin.** Cast shadow, layover and HAND depend on terrain *outside* the grid:
a ridge 1 km up-range casts shadow into it, and a drainage network is defined by its
catchment. For small grids (a 5 km Sen1Floods11 chip) the terrain layers are computed on
the grid expanded by ``terrain_margin_px`` on every side and cropped back.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from floodfusion.config import ExperimentConfig
from floodfusion.data.archive import (
    IngestSources,
    grd_id_for_rtc,
    items_by_id,
    product_annotation_href,
)
from floodfusion.data.chipping import (
    CLEAR_QUALITY,
    MIN_OPTICAL_CLEAR_FRACTION,
    AnalysisGrid,
    extract_chips,
    write_event_stack,
)
from floodfusion.data.encoding import encode_stack
from floodfusion.data.records import ChipRecord, atomic_write_bytes, write_manifest
from floodfusion.data.s1_annotation import S1Annotation, parse_annotation
from floodfusion.data.stac import PROFILES, MatchedScenes, group_by_time

__all__ = [
    "GridReference",
    "GridTask",
    "align_optical_to_sar",
    "assemble_on_grid",
    "circular_mean_deg",
    "expand_grid",
    "incidence_on_grid",
    "label_optical_window_shifts",
    "label_window_shifts",
    "pooled_shift",
    "pooled_shift_ci_halfwidth",
    "qa_verdict",
    "read_acquisition_geometry",
    "sar_optical_window_shifts",
    "shift_pixels",
    "terrain_on_grid",
    "to_split_crs",
    "write_event_basins",
]

LOGGER = logging.getLogger(__name__)

#: Incidence varies by ~0.0006 deg per 10 m pixel: a 16-px lattice + bilinear is exact enough.
_INCIDENCE_LATTICE_PX = 16
_SAR_LAYERS = ("vv_post_db", "vh_post_db", "vv_pre_db", "vh_pre_db")
#: A co-registration check on fewer windows is not a measurement (flat terrain offers the
#: terrain simulation nothing to correlate; one window's peak is noise).
MIN_QA_WINDOWS = 5
#: ...and neither is one whose median shift has a standard error above this (overlap-corrected,
#: see coregistration.median_shift_se). Calibrated on the release data's null case — same-orbit
#: pre/post RTC pairs, which agree by construction: of the 379 Sen1Floods11 grids, those within
#: this bound read >= 1 px in 2 % of cases, the others in 36 %. A 1 px verdict is then >= 2 SE
#: from zero.
QA_MAX_SE_PX = 0.5
#: A pooled pass/event offset counts as a measurement only if its (approximate) 95 % interval —
#: chips resampled, since the windows of one chip are not independent — is at most this wide on
#: each side. The decision it feeds is a whole-pixel shift: an offset pinned to within 1 px picks
#: a shift that beats none in expectation, while a garbage estimate (a few chips, +/-4 px) is
#: refused. (First set to 0.5 px; on Ghana that refused two independent estimates agreeing on
#: ~(1, 1) px, and it was widened before the other events were measured and before any model ran.)
SHIFT_MAX_CI_PX = 1.0
_BOOTSTRAP_RESAMPLES = 1000
#: A systematic SAR-optical misregistration of at least this many pixels is corrected by
#: moving the optical layers onto the SAR/DEM geometry (measured on the real archive:
#: Sentinel-2 L2A of 2018-2019 sits ~2 px from the RTC gamma0; see align_optical_to_sar).
OPTICAL_SHIFT_MIN_PX = 1.0


class AcquisitionGeometry(Protocol):
    """What the assembly needs of a scene's geometry: a GRD annotation or the orbit-model fallback."""

    heading_deg: float
    geolocation: dict[str, np.ndarray]  # longitude, latitude, incidence tie points

    @property
    def look_azimuth_deg(self) -> float: ...

    @property
    def orbit_state(self) -> str: ...


@dataclass(frozen=True)
class GridReference:
    """Reference layers on the grid.

    ``label`` is float: 1 = flood, 0 = not flood, NaN = unlabelled. Permanent water is
    already removed from it (flood = observed water AND NOT permanent water, ``docs/05``
    section 3.3).
    """

    label: np.ndarray
    inside_aoi: np.ndarray
    permanent: np.ndarray
    landcover: np.ndarray


@dataclass(frozen=True)
class GridTask:
    """Everything a route decides; :func:`assemble_on_grid` does the rest."""

    event_id: str
    out_dir: Path
    grid: AnalysisGrid
    matched: MatchedScenes
    annotations: tuple[AcquisitionGeometry, ...]
    reference: GridReference
    basins: Mapping[str, Any]
    dem_items: tuple[Mapping[str, Any], ...]
    extra_meta: Mapping[str, Any] = field(default_factory=dict)
    terrain_margin_px: int = 0
    max_s2_candidates: int = 4
    #: Move optical onto the SAR/DEM geometry when they are measurably misregistered.
    coregister_optical: bool = True
    #: The label was drawn on the optical image (Sen1Floods11): it moves with the optical.
    label_follows_optical: bool = False
    #: A shift measured once for the whole optical pass (pooled over many grids); ``None``
    #: measures it on this grid alone.
    optical_shift_px: tuple[int, int] | None = None
    #: A shift for the label alone, when it was measured not to follow the optical image
    #: (Sen1Floods11 events whose labels were drawn on another geometry). Overrides
    #: ``label_follows_optical``.
    label_shift_px: tuple[int, int] | None = None
    generator: str = "floodfusion.data.ingest (Route B: Planetary Computer sentinel-1-rtc)"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _grid_lonlat(grid: AnalysisGrid, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    a, _, c, _, e, f = grid.transform
    x = c + (cols + 0.5) * a
    y = f + (rows + 0.5) * e
    return Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform(x, y)


def incidence_on_grid(annotations: Sequence[S1Annotation], grid: AnalysisGrid) -> np.ndarray:
    """Per-pixel ellipsoid incidence angle on the grid, from every frame's tie points."""
    from scipy.interpolate import griddata

    from floodfusion.data.preprocessing import interpolate_lut

    lon_pts = np.concatenate([a.geolocation["longitude"] for a in annotations])
    lat_pts = np.concatenate([a.geolocation["latitude"] for a in annotations])
    inc_pts = np.concatenate([a.geolocation["incidence"] for a in annotations])
    lines = np.unique(np.r_[np.arange(0, grid.height, _INCIDENCE_LATTICE_PX), grid.height - 1])
    pixels = np.unique(np.r_[np.arange(0, grid.width, _INCIDENCE_LATTICE_PX), grid.width - 1])
    rr, cc = np.meshgrid(lines, pixels, indexing="ij")
    lon, lat = _grid_lonlat(grid, rr.astype(np.float64), cc.astype(np.float64))
    pts = np.column_stack([lon_pts, lat_pts])
    coarse = griddata(pts, inc_pts, (lon, lat), method="linear")
    hole = ~np.isfinite(coarse)
    if hole.any():
        coarse[hole] = griddata(pts, inc_pts, (lon[hole], lat[hole]), method="nearest")
    return interpolate_lut(coarse, (grid.height, grid.width), lut_lines=lines, lut_pixels=pixels)


def circular_mean_deg(values: Sequence[float]) -> float:
    rad = np.radians(np.asarray(values, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def expand_grid(grid: AnalysisGrid, margin_px: int) -> AnalysisGrid:
    """The same grid grown by ``margin_px`` whole pixels on every side (pixel-aligned)."""
    if margin_px < 0:
        raise ValueError(f"margin_px must be >= 0, got {margin_px}")
    if margin_px == 0:
        return grid
    a, b, c, d, e, f = grid.transform
    return replace(grid, transform=(a, b, c - margin_px * a, d, e, f - margin_px * e),
                   width=grid.width + 2 * margin_px, height=grid.height + 2 * margin_px)


def read_acquisition_geometry(sources: IngestSources, matched: MatchedScenes, *,
                              fallback: bool = True) -> list[AcquisitionGeometry]:
    """The post-event frames' GRD product annotations (heading, tie-point incidence).

    Planetary Computer's ``sentinel-1-grd`` has gaps (an RTC frame whose source GRD is not
    listed). With ``fallback`` the geometry is then modelled from the RTC frame footprints
    and the orbit (:mod:`floodfusion.data.orbit_model`, < 1 deg incidence error) and the
    substitution is logged and recorded (``geometry_source``); without it, it raises.
    """
    frames = matched.s1_post_frames or (matched.s1_post_id,)
    try:  # only a MISSING item or asset falls back; a parse error of a present annotation propagates
        grd_items = items_by_id(sources.client, "sentinel-1-grd", [grd_id_for_rtc(i) for i in frames])
        hrefs = [product_annotation_href(it, "vv") for it in grd_items]
    except LookupError as exc:  # items_by_id: not listed; product_annotation_href: no annotation asset
        if not fallback:
            raise
        LOGGER.warning("%s: no GRD annotation (%s) — geometry from the RTC footprints and the orbit model",
                       matched.event_id, str(exc)[:120])
        return _orbit_model_geometry(sources, matched, frames)
    annotations: list[AcquisitionGeometry] = [parse_annotation(sources.fetch_bytes(h)) for h in hrefs]
    states = {a.orbit_state for a in annotations}
    if states != {matched.orbit_state}:
        raise ValueError(f"annotation pass {states} disagrees with the catalogue ({matched.orbit_state})")
    return annotations


def _orbit_model_geometry(sources: IngestSources, matched: MatchedScenes,
                          frames: Sequence[str]) -> list[AcquisitionGeometry]:
    from shapely.geometry import box, shape

    from floodfusion.data.orbit_model import footprint_geometry

    def footprint(item: Mapping[str, Any]) -> Any:
        if item.get("geometry"):
            return shape(item["geometry"])
        LOGGER.warning("%s has no footprint geometry: using its bbox (incidence less exact)", item["id"])
        return box(*item["bbox"])

    rtc = items_by_id(sources.client, PROFILES["planetary_computer"].s1_collection, frames)
    return [footprint_geometry(footprint(it), matched.orbit_state) for it in rtc]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def _read_sar(sources: IngestSources, matched: MatchedScenes, grid: AnalysisGrid) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from floodfusion.data.sources import read_s1_rtc

    collection = PROFILES["planetary_computer"].s1_collection
    post = items_by_id(sources.client, collection, matched.s1_post_frames or (matched.s1_post_id,))
    pre = items_by_id(sources.client, collection, matched.s1_pre_frames or (matched.s1_pre_id,))
    layers: dict[str, np.ndarray] = {}
    qa: dict[str, Any] = {}
    for when, items in (("post", post), ("pre", pre)):
        for pol in ("VV", "VH"):
            name = f"{pol.lower()}_{when}_db"
            layers[name], qa[name] = read_s1_rtc(items, grid, pol)
    return layers, qa


def _terrain(dem_items: Sequence[Mapping[str, Any]], grid: AnalysisGrid, theta: np.ndarray, phi: float) -> dict[str, np.ndarray]:
    from floodfusion.data.sources import read_dem
    from floodfusion.masks import (
        cast_shadow_and_layover,
        hand_on_dem_support,
        horn_slope_aspect,
        local_incidence_angle,
    )

    dem = read_dem(dem_items, grid)
    holes = ~np.isfinite(dem)
    if holes.all():
        raise ValueError("DEM has no finite values over the AOI")
    if holes.any():
        # Filled with the lowest elevation: a hole becomes a sink, never a ridge that would cast shadow.
        LOGGER.warning("DEM: %.2f %% of the grid has no elevation; filled with the minimum", 100 * holes.mean())
    dem = np.where(holes, np.nanmin(dem), dem)
    px = grid.pixel_size
    slope, aspect = horn_slope_aspect(dem, px, px)
    shadow, layover = cast_shadow_and_layover(dem, theta, phi, px)
    return {
        "dem": dem, "slope": slope, "theta_loc": local_incidence_angle(slope, aspect, theta, phi),
        "shadow": shadow, "layover": layover, "hand": hand_on_dem_support(dem, px), "dem_filled": holes,
    }


def terrain_on_grid(
    dem_items: Sequence[Mapping[str, Any]],
    grid: AnalysisGrid,
    annotations: Sequence[S1Annotation],
    phi: float,
    *,
    margin_px: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Terrain layers and ellipsoid incidence on ``grid``, computed with ``margin_px`` of context.

    Returns ``({"dem", "slope", "theta_loc", "shadow", "layover", "hand"}, theta)``, every
    array of the grid's shape.
    """
    big = expand_grid(grid, margin_px)
    theta_big = incidence_on_grid(annotations, big)
    terrain = _terrain(dem_items, big, theta_big, phi)
    if margin_px == 0:
        return terrain, theta_big
    window = (slice(margin_px, margin_px + grid.height), slice(margin_px, margin_px + grid.width))
    return {k: v[window] for k, v in terrain.items()}, theta_big[window]


@dataclass(frozen=True)
class _OpticalLayers:
    items: tuple[dict[str, Any], ...]
    clear_fraction: float
    reflectance: np.ndarray | None
    quality: np.ndarray | None
    offsets: dict[str, Any] | None

    @property
    def available(self) -> bool:
        return self.reflectance is not None


def _select_optical(
    sources: IngestSources, matched: MatchedScenes, grid: AnalysisGrid, inside_aoi: np.ndarray, max_candidates: int,
) -> tuple[list[dict[str, Any]], float]:
    """The overpass with the largest **AOI** clear fraction (from SCL), or ``([], 0)``."""
    from floodfusion.data.sources import read_s2_scl
    from floodfusion.masks import quality_from_scl

    if not matched.s2_ids:
        return [], 0.0
    items = items_by_id(sources.client, "sentinel-2-l2a", matched.s2_ids)
    if len(matched.s2_datetimes) == len(items):
        passes = [[items[i] for i in group] for group in group_by_time(matched.s2_datetimes)]
    else:
        passes = [[item] for item in items]
    def scene_cloud(group: Sequence[Mapping[str, Any]]) -> float:  # unknown cover ranks last, deterministically
        values = [float(i["properties"].get("eo:cloud_cover", math.nan)) for i in group]
        finite = [v for v in values if math.isfinite(v)]
        return float(np.mean(finite)) if finite else math.inf

    ranked = sorted(passes, key=lambda p: (scene_cloud(p), p[0]["id"]))
    best: tuple[list[dict[str, Any]], float] = ([], 0.0)
    for group in ranked[:max_candidates]:
        quality = quality_from_scl(read_s2_scl(group, grid))
        clear = float((quality[inside_aoi] >= CLEAR_QUALITY).mean()) if inside_aoi.any() else 0.0
        LOGGER.info("S2 pass %s: AOI clear fraction %.1f %%", group[0]["id"], 100 * clear)
        if clear > best[1]:
            best = (group, clear)
    return best


def _optical_layers(sources: IngestSources, task: GridTask) -> _OpticalLayers:
    from floodfusion.data.sources import read_s2_reflectance
    from floodfusion.masks import quality_from_scl

    items, clear = _select_optical(sources, task.matched, task.grid, task.reference.inside_aoi, task.max_s2_candidates)
    if not items or clear < MIN_OPTICAL_CLEAR_FRACTION:
        LOGGER.info("%s: no usable Sentinel-2 overpass (best AOI clear fraction %.1f %%): SAR-only",
                    task.event_id, 100 * clear)
        return _OpticalLayers(tuple(items), clear, None, None, None)
    rho, scl, offsets = read_s2_reflectance(items, task.grid)
    return _OpticalLayers(tuple(items), clear, rho, quality_from_scl(scl), offsets)


def shift_pixels(array: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    """``out[..., r, c] = array[..., r + dy, c + dx]``; pixels with no source get ``fill``.

    Whole pixels only: the layer is moved, never resampled a second time.
    """
    out = np.full_like(array, fill)
    h, w = array.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    dst_r, src_r = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
    dst_c, src_c = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
    out[..., dst_r, dst_c] = array[..., src_r, src_c]
    return out


def pooled_shift(per_grid_shifts: Sequence[np.ndarray], *,
                 min_shift_px: float = OPTICAL_SHIFT_MIN_PX) -> dict[str, Any]:
    """One whole-pixel shift from the window shifts of many grids of one pass (or event).

    The median over **all windows** (not the median of per-grid medians), so a grid with
    few textured windows weighs little. ``shift_px`` is ``[0, 0]`` below ``min_shift_px``
    or with fewer than :data:`MIN_QA_WINDOWS` windows in total. ``ci_halfwidth_px`` is the
    precision of the median (:func:`pooled_shift_ci_halfwidth`); deciding whether it is
    good enough is the caller's choice.
    """
    shifts = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in per_grid_shifts]
    allw = np.concatenate(shifts) if shifts else np.empty((0, 2))
    out: dict[str, Any] = {"n_windows": int(len(allw)), "n_grids": sum(1 for s in shifts if len(s)),
                           "median_shift": [float("nan"), float("nan")], "shift_px": [0, 0],
                           "ci_halfwidth_px": pooled_shift_ci_halfwidth(shifts)}
    if len(allw) < MIN_QA_WINDOWS:
        return out
    dy, dx = np.median(allw, axis=0)
    out["median_shift"] = [float(dy), float(dx)]
    if np.hypot(dy, dx) >= min_shift_px:
        out["shift_px"] = [_round_half_up(dy), _round_half_up(dx)]
    return out


def _round_half_up(value: float) -> int:
    """``round`` is banker's rounding (2.5 -> 2, 3.5 -> 4): a shift must not depend on parity."""
    return int(math.floor(float(value) + 0.5))


def pooled_shift_ci_halfwidth(per_grid_shifts: Sequence[np.ndarray], *, n_boot: int = _BOOTSTRAP_RESAMPLES,
                              seed: int = 0) -> float:
    """Half-width (px, worse axis) of the 95 % interval of the pooled median shift.

    Grids are resampled with replacement — the windows of one grid overlap and share its
    content, so the grid is the independent unit. Chips that disagree (half say 0 px, half
    4 px) give a wide interval however many windows each has. ``nan`` without windows.
    """
    grids = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in per_grid_shifts]
    grids = [g for g in grids if len(g)]
    if len(grids) < 2:  # one grid resampled is always itself: an interval of 0 that means nothing
        return float("nan")
    rng = np.random.default_rng(seed)
    medians = np.array([np.median(np.concatenate([grids[i] for i in rng.integers(0, len(grids), len(grids))]), axis=0)
                        for _ in range(n_boot)])
    lo, hi = np.percentile(medians, [2.5, 97.5], axis=0)
    return float(np.max((hi - lo) / 2.0))


def label_optical_window_shifts(reflectance: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Per-window ``(dy, dx)`` of a water label relative to the optical SWIR, on clear pixels.

    No SAR involved: whether a label was drawn on this optical image is measured directly.
    """
    from floodfusion.data.coregistration import median_displacement, structure_image

    lab = np.asarray(label, dtype=np.float64)
    swir = np.asarray(reflectance[_ALIGNMENT_BAND], dtype=np.float64)
    valid = np.isfinite(lab) & np.isfinite(swir)
    if not valid.any():
        return np.empty((0, 2))
    swir = np.where(valid, swir, float(np.median(swir[valid])))
    water = np.where(np.isfinite(lab), lab, 0.0)
    return median_displacement(structure_image(swir, log=True), structure_image(water, log=False),
                               valid=valid)["shifts"]


def label_window_shifts(sar_post_db: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Per-window ``(dy, dx)`` of a water label (1 water, 0 land, NaN unlabelled) relative to SAR.

    Its water edges against the SAR's: the edges a flood label is drawn along, and the
    sharpest structure C-band has. Windows less than 90 % labelled are skipped.
    """
    from floodfusion.data.coregistration import median_displacement, structure_image

    lab = np.asarray(label, dtype=np.float64)
    sar = np.asarray(sar_post_db, dtype=np.float64)
    valid = np.isfinite(lab) & np.isfinite(sar)
    if not valid.any():
        return np.empty((0, 2))
    post = np.nan_to_num(sar, nan=float(np.nanmedian(sar[valid])))
    water = np.where(np.isfinite(lab), lab, 0.0)
    return median_displacement(structure_image(post, log=False), structure_image(water, log=False),
                               valid=valid)["shifts"]


def sar_optical_window_shifts(sar_post_db: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """Per-window ``(dy, dx)`` of optical SWIR relative to SAR (input of :func:`pooled_shift`).

    NaN reflectance (cloud) does not vote: windows less than 90 % clear are skipped.
    """
    return _sar_optical_offset(sar_post_db, reflectance)["shifts"]


#: Reflectance band used to register optical to SAR: B11 (SWIR). Its strongest edges are
#: water/land, as in C-band; NIR edges also follow vegetation vigour, which SAR does not see,
#: and biased the estimate by up to 0.7 px against the label edges on Sen1Floods11.
_ALIGNMENT_BAND = 2


def _sar_optical_offset(sar_post_db: np.ndarray, reflectance: np.ndarray,
                        quality: np.ndarray | None = None) -> dict[str, Any]:
    from floodfusion.data.coregistration import median_displacement, structure_image

    swir = np.asarray(reflectance[_ALIGNMENT_BAND], dtype=np.float64)
    valid = np.isfinite(swir) & np.isfinite(sar_post_db)
    if quality is not None:
        valid &= np.asarray(quality) >= CLEAR_QUALITY
    if not valid.any():
        return median_displacement(np.zeros((1, 1)), np.zeros((1, 1)))
    post = np.nan_to_num(sar_post_db, nan=float(np.nanmedian(sar_post_db)))
    swir = np.where(valid, swir, float(np.median(swir[valid])))
    return median_displacement(structure_image(post, log=False), structure_image(swir, log=True), valid=valid)


def align_optical_to_sar(
    sar_post_db: np.ndarray,
    reflectance: np.ndarray,
    quality: np.ndarray,
    *,
    label: np.ndarray | None = None,
    inside_aoi: np.ndarray | None = None,
    min_shift_px: float = OPTICAL_SHIFT_MIN_PX,
    fixed_shift_px: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Move optical (and, if given, a label drawn on it) onto the SAR geometry.

    The SAR/DEM geometry is the reference: RTC gamma0 is geocoded with precise orbits and
    the same DEM as every terrain layer. The systematic offset is the median shift vector
    over many clear windows of SAR and SWIR (B11) structure images; it is applied in **whole
    pixels** when its length reaches ``min_shift_px`` and it is a measurement (standard error
    at most :data:`QA_MAX_SE_PX`). Sub-pixel residuals stay — moving by a fraction would
    resample the optical a second time.

    ``fixed_shift_px`` applies a shift measured elsewhere — pooled over every grid of one
    optical pass, which is where the error lives — and only reports this grid's own estimate.

    Returns ``(reflectance, quality, label, inside_aoi, info)``; vacated pixels become NaN
    reflectance, quality 0 (not clear), NaN label and outside the AOI.
    """
    measured = _sar_optical_offset(sar_post_db, reflectance, quality)
    info: dict[str, Any] = {"measured_shift_px": [measured["median_dy"], measured["median_dx"]],
                            "measured_se_px": measured["median_se_px"],
                            "n_windows": measured["n_windows"], "applied_shift_px": [0, 0],
                            "label_shifted": False, "source": "this grid" if fixed_shift_px is None else "pooled"}
    if fixed_shift_px is not None:
        dy, dx = int(fixed_shift_px[0]), int(fixed_shift_px[1])
        if (dy, dx) == (0, 0):
            return reflectance, quality, label, inside_aoi, info
    elif (measured["n_windows"] < MIN_QA_WINDOWS or not measured["median_se_px"] <= QA_MAX_SE_PX
          or not measured["median_shift_px"] >= min_shift_px):
        return reflectance, quality, label, inside_aoi, info
    else:
        dy, dx = round(measured["median_dy"]), round(measured["median_dx"])
    info["applied_shift_px"] = [dy, dx]
    reflectance = shift_pixels(reflectance, dy, dx, np.nan)
    quality = shift_pixels(quality, dy, dx, 0.0)
    if label is not None:
        label = shift_pixels(label, dy, dx, np.nan)
        info["label_shifted"] = True
    if inside_aoi is not None:
        inside_aoi = shift_pixels(inside_aoi, dy, dx, False)
    LOGGER.info("optical moved by (%d, %d) px onto the SAR geometry (%s; this grid: %.2f, %.2f over %d windows)",
                dy, dx, info["source"], measured["median_dy"], measured["median_dx"], measured["n_windows"])
    return reflectance, quality, label, inside_aoi, info


def _coregister(task: GridTask, sar: Mapping[str, np.ndarray], optical: _OpticalLayers,
                ref: GridReference) -> tuple[_OpticalLayers, GridReference, dict[str, Any]]:
    if not task.coregister_optical:
        return optical, ref, {"applied_shift_px": [0, 0], "skipped": "off"}
    if task.label_shift_px is not None:  # the label has its own geometry: move it on its own
        dy, dx = (int(v) for v in task.label_shift_px)
        optical, ref, info = _coregister(replace(task, label_shift_px=None, label_follows_optical=False),
                                         sar, optical, ref)
        if (dy, dx) != (0, 0):
            ref = replace(ref, label=shift_pixels(ref.label, dy, dx, np.nan),
                          inside_aoi=shift_pixels(ref.inside_aoi, dy, dx, False))
        return optical, ref, {**info, "label_shift_px": [dy, dx], "label_shifted": (dy, dx) != (0, 0)}
    follows = task.label_follows_optical
    if not optical.available:
        # A label drawn on the optical pass is offset like that pass even where this grid's
        # optical is too cloudy to use: a pooled (pass-level) shift still moves it, or a
        # cloudy chip would keep its label 1-2 px off the SAR/DEM it is scored against.
        if follows and task.optical_shift_px is not None and tuple(task.optical_shift_px) != (0, 0):
            dy, dx = (int(v) for v in task.optical_shift_px)
            ref = replace(ref, label=shift_pixels(ref.label, dy, dx, np.nan),
                          inside_aoi=shift_pixels(ref.inside_aoi, dy, dx, False))
            return optical, ref, {"applied_shift_px": [dy, dx], "label_shifted": True, "source": "pooled",
                                  "skipped": "no usable optical on this grid; label moved by the pass shift"}
        return optical, ref, {"applied_shift_px": [0, 0], "skipped": "no optical"}
    rho, quality, label, inside, info = align_optical_to_sar(
        sar["vv_post_db"], optical.reflectance, optical.quality,
        label=ref.label if follows else None, inside_aoi=ref.inside_aoi if follows else None,
        fixed_shift_px=task.optical_shift_px)
    if info["applied_shift_px"] == [0, 0]:
        return optical, ref, info
    optical = replace(optical, reflectance=rho, quality=quality)
    if follows:
        ref = replace(ref, label=label, inside_aoi=inside)
    return optical, ref, info


def qa_verdict(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mark each check measurable or not and give the ``P1-M1`` verdict over the measurable ones.

    ``p1_m1_pass`` is ``True`` when every measurable check is below 1 px, ``False`` when one is
    not, and ``None`` when nothing could be measured (flat, textureless or cloud-covered grids):
    "not measured" is not "failed".
    """
    out: dict[str, Any] = {}
    measured = []
    for name, check in checks.items():
        ok = check["n_windows"] >= MIN_QA_WINDOWS and check.get("median_se_px", float("nan")) <= QA_MAX_SE_PX
        out[name] = {**check, "measurable": bool(ok)}
        if ok:
            measured.append(check["median_shift_px"])
    out["p1_m1_pass"] = None if not measured else all(m < 1.0 for m in measured)
    return out


def _coregistration_qa(sar: Mapping[str, np.ndarray], terrain: Mapping[str, np.ndarray],
                       reflectance: np.ndarray | None, quality: np.ndarray | None = None) -> dict[str, Any]:
    """``P1-M1``: median displacements (pixels) between layers that must be co-registered."""
    from floodfusion.data.coregistration import (
        median_displacement,
        structure_image,
        validate_geocoding,
    )

    def summary(result: Mapping[str, Any]) -> dict[str, float]:
        return {k: float(v) if isinstance(v, int | float) else v for k, v in result.items() if k != "shifts"}

    post = np.nan_to_num(sar["vv_post_db"], nan=float(np.nanmedian(sar["vv_post_db"])))
    pre = np.nan_to_num(sar["vv_pre_db"], nan=float(np.nanmedian(sar["vv_pre_db"])))
    qa = {
        "pre_vs_post": summary(median_displacement(
            structure_image(pre, log=False), structure_image(post, log=False),
            valid=np.isfinite(sar["vv_pre_db"]) & np.isfinite(sar["vv_post_db"]))),
        "terrain_simulation_vs_post": summary(validate_geocoding(
            post, terrain["theta_loc"], shadow=terrain["shadow"], layover=terrain["layover"])),
    }
    if reflectance is not None:
        # The alignment's own band and cloud mask: a per-grid check of what was applied.
        qa["sar_vs_optical_swir"] = summary(_sar_optical_offset(sar["vv_post_db"], reflectance, quality))
    return qa_verdict(qa)


def _grid_meta(task: GridTask, ref: GridReference, phi: float, theta: np.ndarray, optical: _OpticalLayers,
               layers: Mapping[str, np.ndarray], validity: np.ndarray) -> dict[str, Any]:
    from floodfusion.utils.env import git_state

    valid = validity == 0
    grid, matched, label = task.grid, task.matched, np.nan_to_num(ref.label, nan=0.0) > 0.5
    return {
        "event_id": task.event_id, "epsg": grid.epsg, "transform": list(grid.transform),
        "width": grid.width, "height": grid.height, "grid_notes": list(grid.notes),
        "relative_orbit": matched.relative_orbit, "orbit_state": matched.orbit_state,
        "look_azimuth_deg": phi, "heading_deg": [a.heading_deg for a in task.annotations],
        "geometry_source": sorted({getattr(a, "source", "GRD product annotation") for a in task.annotations}),
        "incidence_deg": [float(np.nanmin(theta)), float(np.nanmax(theta))],
        "s1_post_frames": list(matched.s1_post_frames), "s1_pre_frames": list(matched.s1_pre_frames),
        "s1_post_datetime": matched.s1_post_datetime, "s1_pre_datetime": matched.s1_pre_datetime,
        "has_optical": optical.available,
        "s2_items": [i["id"] for i in optical.items] if optical.available else [],
        "s2_aoi_clear_fraction": optical.clear_fraction, "s2_offsets": optical.offsets,
        "dem_ids": [str(i["id"]) for i in task.dem_items], "terrain_margin_px": task.terrain_margin_px,
        "flood_fraction_of_valid": float((label & valid).sum() / max(int(valid.sum()), 1)),
        "permanent_water_fraction": float(ref.permanent.mean()),
        "shadow_fraction": float(layers["shadow"].mean()), "layover_fraction": float(layers["layover"].mean()),
        "outside_aoi_fraction": float((~ref.inside_aoi).mean()),
        "nodata_fraction": float(layers["nodata"].mean()),
        "dem_filled_fraction": float(layers["dem_filled"].mean()) if "dem_filled" in layers else 0.0,
        **task.extra_meta, **git_state(),
    }


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------
def assemble_on_grid(task: GridTask, cfg: ExperimentConfig, sources: IngestSources, *, split_epsg: int) -> dict[str, Any]:
    """Build ``<out_dir>/{stack.tif, chips.parquet, basins.gpkg, event.json}``; return the metadata.

    ``event.json`` is written **last**: its presence means the whole assembly, QA included,
    finished — the resumable drivers skip on it.
    """
    from floodfusion.masks import build_validity_mask

    grid, ref = task.grid, task.reference
    task.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("event.json", "chips.parquet", "basins.gpkg"):  # a re-run must never mix with an old one
        (task.out_dir / stale).unlink(missing_ok=True)
    sar, rtc_qa = _read_sar(sources, task.matched, grid)
    phi = circular_mean_deg([a.look_azimuth_deg for a in task.annotations])
    terrain, theta = terrain_on_grid(task.dem_items, grid, task.annotations, phi, margin_px=task.terrain_margin_px)
    optical, ref, alignment = _coregister(task, sar, _optical_layers(sources, task), ref)

    nodata = ~np.logical_and.reduce([np.isfinite(sar[k]) for k in _SAR_LAYERS])
    validity = build_validity_mask(terrain["layover"], terrain["shadow"], ref.permanent, nodata, None,
                                   slope_max_deg=None, outside_aoi=~ref.inside_aoi)
    stack = encode_stack(
        vv_post_db=sar["vv_post_db"], vh_post_db=sar["vh_post_db"], vv_pre_db=sar["vv_pre_db"],
        vh_pre_db=sar["vh_pre_db"], reflectance=optical.reflectance, slope_deg=terrain["slope"],
        hand_m=terrain["hand"], local_incidence_deg=terrain["theta_loc"],
        label=np.where(ref.inside_aoi, ref.label.astype(np.float64), np.nan),
        validity=validity, optical_quality=optical.quality, landcover=ref.landcover,
        grid_cfg=cfg.grid, norm=cfg.norm,
    )
    matched = task.matched
    stack_path = write_event_stack(task.out_dir / "stack.tif", stack, grid, nodata=cfg.grid.nodata_int16, tags={
        "event_id": task.event_id, "s1_post": matched.s1_post_id, "s1_pre": matched.s1_pre_id,
        "s2": ",".join(i["id"] for i in optical.items) if optical.available else "",
        "look_azimuth_deg": f"{phi:.3f}", "generator": task.generator,
    })
    records, counts = extract_chips(
        stack_path, event_id=task.event_id, basins=task.basins, data_cfg=cfg.data, grid_cfg=cfg.grid,
        relative_orbit=matched.relative_orbit, orbit_state=matched.orbit_state, has_optical=optical.available,
    )
    records = to_split_crs(records, grid.epsg, split_epsg)
    if records:
        write_event_basins(task.out_dir / "basins.gpkg", task.basins, grid.epsg, split_epsg, task.event_id)
        write_manifest(records, task.out_dir / "chips.parquet",
                       provenance={"event_id": task.event_id, "split_epsg": split_epsg})
    else:  # no window passed the valid-fraction filter: nothing to train or score on this grid
        LOGGER.warning("%s: no chip passed the window filters %s; no manifest written", task.event_id, counts)
    meta = {
        **_grid_meta(task, ref, phi, theta, optical, {**terrain, "nodata": nodata}, validity),
        "chip_counts": counts, "n_chips": len(records), "basins": sorted(task.basins), "rtc_qa": rtc_qa,
        "coregistration_qa": _coregistration_qa(sar, terrain, optical.reflectance, optical.quality),
        "optical_alignment": alignment,
    }
    atomic_write_bytes(task.out_dir / "event.json",
                       json.dumps(_json_safe(meta), indent=2, default=str, allow_nan=False).encode("utf-8"))
    verdict = meta["coregistration_qa"]["p1_m1_pass"]
    if verdict is False:
        LOGGER.warning("%s: P1-M1 co-registration check did not pass: %s", task.event_id, meta["coregistration_qa"])
    elif verdict is None:
        LOGGER.info("%s: P1-M1 co-registration not measurable on this grid (no precise check)", task.event_id)
    return meta


def _json_safe(value: Any) -> Any:
    """NaN/inf become null: ``event.json`` must be valid JSON for any reader, not just Python's."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def to_split_crs(records: Sequence[ChipRecord], src_epsg: int, dst_epsg: int) -> list[ChipRecord]:
    """Chip centroids into the dataset-level metric CRS of the split (``docs/05`` 6.4)."""
    from pyproj import Transformer

    if src_epsg == dst_epsg or not records:
        return list(records)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    xs, ys = tr.transform([r.centroid_x for r in records], [r.centroid_y for r in records])
    return [replace(r, centroid_x=float(x), centroid_y=float(y)) for r, x, y in zip(records, xs, ys)]


def write_event_basins(path: Path, basins: Mapping[str, Any], src_epsg: int, dst_epsg: int, event_id: str) -> None:
    import geopandas as gpd

    frame = gpd.GeoDataFrame({"basin_id": list(basins), "event_id": [event_id] * len(basins)},
                             geometry=list(basins.values()), crs=f"EPSG:{src_epsg}").to_crs(epsg=dst_epsg)
    tmp = path.with_name(f".tmp.{path.name}")
    frame.to_file(tmp, driver="GPKG", layer="basins")
    tmp.replace(path)
