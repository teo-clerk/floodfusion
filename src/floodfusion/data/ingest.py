r"""Real-data ingest: one flood event from public archives to an analysis-ready chip set.

``P1-M0`` + ``P1-M1``. Everything credential-free through Microsoft Planetary Computer
(anonymous SAS signing):

1. **Match** a pre/post Sentinel-1 pair on the same relative orbit and pass
   (:func:`floodfusion.data.stac.search_matched_pairs`; raises rather than fall back).
2. **Grid**: the single 10 m UTM analysis grid of the AOI
   (:func:`floodfusion.data.chipping.build_analysis_grid`).
3. **SAR** — Route B: gamma0 from ``sentinel-1-rtc`` (already terrain-flattened), every
   frame of each acquisition mosaicked, resampled once in linear power; the RTC checklist
   of ``docs/04`` section 9 is applied to every layer.
4. **Acquisition geometry** from the matching ``sentinel-1-grd`` product annotation:
   per-scene heading (so the look azimuth is computed, not assumed) and the per-pixel
   ellipsoid incidence angle from the geolocation tie points.
5. **Terrain**: Copernicus GLO-30 onto the grid; Horn slope/aspect in the metric CRS,
   local incidence angle, the ray-swept shadow/layover masks, HAND at the 30 m support.
6. **Optical**: the least-cloudy Sentinel-2 L2A overpass in the flood window, judged by
   the **AOI-level** clear fraction from its SCL (not the scene ``eo:cloud_cover``), all
   tiles of that pass mosaicked with per-tile offset harmonisation.
7. **Exclusions**: JRC GSW occurrence >= 80 % (permanent water), the CEMS AOI, nodata.
8. **Label**: the CEMS (or any) observed-flood vector, rasterised on the grid, minus
   permanent water — flood = observed water AND NOT permanent water (``docs/05`` 3.3).
9. **Stack, chips, manifest**; per-event QA (RTC checklist, pre/post and SAR-optical
   co-registration — ``P1-M1``'s median displacement < 1 px — and the terrain
   simulation check).

Inputs the archives cannot provide are named in the event YAML
(``configs/events/example.yaml``): the reference flood delineation (download the CEMS
rapid-mapping vector product of the activation), optionally its AOI, and optionally
HydroBASINS polygons (hydrosheds.org) for basin-blocked splits.

Chip centroids and basin polygons are stored in one dataset-level metric CRS
(``--split-epsg``, default EPSG:3035 ETRS89-LAEA Europe) because events of one dataset sit
in different UTM zones while the buffer test of ``docs/05`` 6.4 needs one metric space;
each event stack keeps its own UTM grid.

Memory scales at roughly 200 B per grid pixel: a 30 x 30 km AOI (9 Mpx) needs ~2 GB.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.chipping import (
    CLEAR_QUALITY,
    MIN_OPTICAL_CLEAR_FRACTION,
    AnalysisGrid,
    build_analysis_grid,
    extract_chips,
    write_event_stack,
)
from floodfusion.data.encoding import encode_stack
from floodfusion.data.records import ChipRecord, atomic_write_bytes, load_manifest, write_manifest
from floodfusion.data.s1_annotation import S1Annotation, parse_annotation
from floodfusion.data.stac import (
    PROFILES,
    EventQuery,
    MatchedScenes,
    open_catalogue,
    profile_for,
    redact,
    save_matched_scenes,
    search_matched_pairs,
)

__all__ = [
    "PC_STAC_URL",
    "EventConfig",
    "IngestSources",
    "build_dataset",
    "grd_id_for_rtc",
    "incidence_on_grid",
    "ingest_event",
    "load_event_config",
    "main",
    "product_annotation_href",
    "rasterise_vector",
]

LOGGER = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_SPLIT_EPSG = 3035
#: Incidence varies by ~0.0006 deg per 10 m pixel: a 16-px lattice + bilinear is exact enough.
_INCIDENCE_LATTICE_PX = 16
_HTTP_TIMEOUT_S = 120
_MAX_FETCH_BYTES = 64 * 1024 * 1024
#: ~200 B per grid pixel in memory: 5e7 px is ~10 GB — a mistyped bbox, not an AOI.
MAX_GRID_PIXELS = 50_000_000
#: GDAL's own HTTP reads (COGs on blob storage) need bounds too.
_GDAL_ENV = {"GDAL_HTTP_TIMEOUT": 60, "GDAL_HTTP_MAX_RETRY": 3, "GDAL_HTTP_RETRY_DELAY": 2,
             "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}


# ---------------------------------------------------------------------------
# Event configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventConfig:
    """One flood event to ingest (``configs/events/*.yaml``).

    Paths are resolved relative to the YAML file. ``label_where`` keeps only features whose
    attribute matches, e.g. ``{"event_type": ["5-Flood"]}`` for CEMS ``observedEventA``.
    """

    event_id: str
    bbox: tuple[float, float, float, float]
    flood_start: date
    flood_end: date
    label_path: Path
    label_layer: str | None = None
    label_where: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    aoi_path: Path | None = None
    aoi_layer: str | None = None
    reference_water_path: Path | None = None
    reference_water_layer: str | None = None
    basins_path: Path | None = None
    basins_layer: str | None = None
    basin_id_field: str = "HYBAS_ID"
    pre_event_window_days: int = 60
    catalogue_url: str = PC_STAC_URL
    max_s2_candidates: int = 4
    notes: str = ""

    def query(self) -> EventQuery:
        return EventQuery(
            event_id=self.event_id, bbox=self.bbox,
            flood_start=datetime.combine(self.flood_start, datetime.min.time()),
            flood_end=datetime.combine(self.flood_end, datetime.min.time()),
            pre_event_window_days=self.pre_event_window_days,
        )


_PATH_FIELDS = {"label_path", "aoi_path", "reference_water_path", "basins_path"}


def load_event_config(path: Path | str) -> EventConfig:
    """Parse and validate an event YAML; every unknown key is reported at once."""
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(EventConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown event keys {unknown}; known: {sorted(known)}")
    missing = sorted({"event_id", "bbox", "flood_start", "flood_end", "label_path"} - set(raw))
    if missing:
        raise ValueError(f"{path}: missing required event keys {missing}")
    values = dict(raw)
    for key in _PATH_FIELDS & set(values):
        if values[key] is not None:
            p = Path(values[key])
            values[key] = p if p.is_absolute() else (path.parent / p).resolve()
    bbox = tuple(float(v) for v in values["bbox"])
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"{path}: bbox must be [west, south, east, north], got {values['bbox']}")
    values["bbox"] = bbox
    for key in ("flood_start", "flood_end"):
        v = values[key]
        values[key] = v if isinstance(v, date) else date.fromisoformat(str(v))
    if values["flood_end"] < values["flood_start"]:
        raise ValueError(f"{path}: flood_end precedes flood_start")
    values["label_where"] = {k: tuple(v) if isinstance(v, list | tuple) else (v,)
                             for k, v in (values.get("label_where") or {}).items()}
    return EventConfig(**values)


# ---------------------------------------------------------------------------
# Archive access (injectable for offline tests)
# ---------------------------------------------------------------------------
def _default_fetch(href: str) -> bytes:
    """Bytes of a local file or an allow-listed, signed HTTPS href (no redirects, size-capped)."""
    from floodfusion.data.sources import check_href, signed

    href = check_href(href)
    if not href.startswith("https://"):
        path = Path(href.removeprefix("file://"))
        if not path.is_file() or path.stat().st_size > _MAX_FETCH_BYTES:
            raise ValueError(f"{path}: not a regular file below {_MAX_FETCH_BYTES} bytes")
        return path.read_bytes()
    import requests

    url = signed(href)
    with requests.get(url, timeout=(10, _HTTP_TIMEOUT_S), allow_redirects=False, stream=True) as response:
        if response.status_code != 200:
            raise OSError(f"HTTP {response.status_code} for {redact(url)}")
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=1 << 20):
            size += len(chunk)
            if size > _MAX_FETCH_BYTES:
                raise OSError(f"{redact(url)}: larger than {_MAX_FETCH_BYTES} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class IngestSources:
    """The catalogue client and the byte fetcher; both replaceable offline."""

    client: Any
    fetch_bytes: Callable[[str], bytes] = _default_fetch

    @classmethod
    def open(cls, catalogue_url: str = PC_STAC_URL) -> IngestSources:
        return cls(client=open_catalogue(catalogue_url))


def _as_dict(item: Any) -> dict[str, Any]:
    return dict(item) if isinstance(item, Mapping) else item.to_dict(include_self_link=False, transform_hrefs=False)


def _items_by_id(client: Any, collection: str, ids: Sequence[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    result = client.search(collections=[collection], ids=list(ids), max_items=len(ids) + 5)
    found = {d["id"]: d for d in (_as_dict(i) for i in (result.items() if hasattr(result, "items") else result))}
    missing = [i for i in ids if i not in found]
    if missing:
        raise LookupError(f"items not found in {collection}: {missing}")
    return [found[i] for i in ids]


def _items_in_bbox(client: Any, collection: str, bbox: Sequence[float]) -> list[dict[str, Any]]:
    result = client.search(collections=[collection], bbox=list(bbox), max_items=200)
    return [_as_dict(i) for i in (result.items() if hasattr(result, "items") else result)]


def grd_id_for_rtc(rtc_id: str) -> str:
    """Planetary Computer names an RTC item after its source GRD product plus ``_rtc``."""
    return rtc_id[:-4] if rtc_id.endswith("_rtc") else rtc_id


def product_annotation_href(grd_item: Mapping[str, Any], polarisation: str = "vv") -> str:
    """Href of the GRD **product** annotation (``annotation/<...>.xml``).

    Planetary Computer's ``schema-product-<pol>`` asset points at the **RFI** annotation
    (``annotation/rfi/rfi-iw-<pol>.xml``), not the product annotation; the product
    annotation sits one directory up as ``annotation/iw-<pol>.xml`` (verified on
    S1A_IW_GRDH_1SDV_20231107T170700_..._062A12).
    """
    assets = grd_item.get("assets", {})
    pol = polarisation.lower()
    for key in (f"annotation-{pol}", f"product-annotation-{pol}"):
        if key in assets:
            return str(assets[key]["href"])
    key = f"schema-product-{pol}"
    if key not in assets:
        raise KeyError(f"GRD item {grd_item.get('id')} has no {key} asset")
    href = str(assets[key]["href"])
    base, sep, query = href.partition("?")
    if "/annotation/rfi/rfi-" in base:
        base = base.replace("/annotation/rfi/rfi-", "/annotation/")
    return base + sep + query


# ---------------------------------------------------------------------------
# Geometry and rasterisation helpers
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


def _circular_mean_deg(values: Sequence[float]) -> float:
    rad = np.radians(np.asarray(values, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def _read_vector(path: Path, layer: str | None, epsg: int, where: Mapping[str, tuple[Any, ...]] | None = None) -> Any:
    import geopandas as gpd

    frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"{path}: vector has no CRS; refusing to guess")
    for column, allowed in (where or {}).items():
        if column not in frame.columns:
            raise KeyError(f"{path}: filter column {column!r} not in {list(frame.columns)}")
        frame = frame[frame[column].isin(allowed)]
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    return frame.to_crs(epsg=epsg)


def rasterise_vector(
    path: Path,
    grid: AnalysisGrid,
    *,
    layer: str | None = None,
    where: Mapping[str, tuple[Any, ...]] | None = None,
) -> np.ndarray:
    """Polygons -> boolean mask on the grid (pixel centres inside; ``all_touched=False``).

    ``all_touched`` would dilate every delineation by up to a pixel — a systematic positive
    bias of the label at every flood edge.
    """
    from rasterio.features import rasterize

    frame = _read_vector(path, layer, grid.epsg, where)
    if frame.empty:
        LOGGER.warning("%s: no features after filtering; rasterised mask is empty", path)
        return np.zeros((grid.height, grid.width), dtype=bool)
    mask = rasterize(((g, 1) for g in frame.geometry), out_shape=(grid.height, grid.width),
                     transform=grid.affine, fill=0, all_touched=False, dtype="uint8")
    return mask.astype(bool)


def _grid_box(grid: AnalysisGrid) -> Any:
    from shapely.geometry import box

    return box(*grid.bounds)


def _event_basins(event: EventConfig, grid: AnalysisGrid) -> dict[str, Any]:
    """``basin_id -> polygon`` (grid CRS) intersecting the AOI; one AOI basin without HydroBASINS."""
    if event.basins_path is None:
        LOGGER.warning("%s: no basins_path — the whole AOI is one basin, so leave-one-basin-out "
                       "degenerates to leave-one-event-out for this event", event.event_id)
        return {f"{event.event_id}_aoi": _grid_box(grid)}
    frame = _read_vector(event.basins_path, event.basins_layer, grid.epsg)
    if event.basin_id_field not in frame.columns:
        raise KeyError(f"{event.basins_path}: no {event.basin_id_field!r} column")
    frame = frame[frame.intersects(_grid_box(grid))]
    if frame.empty:
        raise ValueError(f"{event.event_id}: no basin polygon intersects the AOI")
    # HydroBASINS ids are global: the same basin seen in two events is ONE split group.
    return {f"HB{int(v)}" if isinstance(v, int | float) and float(v).is_integer() else f"HB{v}": geom
            for v, geom in zip(frame[event.basin_id_field], frame.geometry)}


# ---------------------------------------------------------------------------
# The ingest
# ---------------------------------------------------------------------------
def _read_sar(sources: IngestSources, matched: MatchedScenes, grid: AnalysisGrid) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from floodfusion.data.sources import read_s1_rtc

    collection = PROFILES["planetary_computer"].s1_collection
    post = _items_by_id(sources.client, collection, matched.s1_post_frames or (matched.s1_post_id,))
    pre = _items_by_id(sources.client, collection, matched.s1_pre_frames or (matched.s1_pre_id,))
    layers: dict[str, np.ndarray] = {}
    qa: dict[str, Any] = {}
    for when, items in (("post", post), ("pre", pre)):
        for pol in ("VV", "VH"):
            name = f"{pol.lower()}_{when}_db"
            layers[name], qa[name] = read_s1_rtc(items, grid, pol)
    return layers, qa


def _read_geometry(sources: IngestSources, matched: MatchedScenes) -> list[S1Annotation]:
    grd_ids = [grd_id_for_rtc(i) for i in (matched.s1_post_frames or (matched.s1_post_id,))]
    grd_items = _items_by_id(sources.client, "sentinel-1-grd", grd_ids)
    annotations = [parse_annotation(sources.fetch_bytes(product_annotation_href(it, "vv"))) for it in grd_items]
    states = {a.orbit_state for a in annotations}
    if states != {matched.orbit_state}:
        raise ValueError(f"annotation pass {states} disagrees with the catalogue ({matched.orbit_state})")
    return annotations


def _select_optical(
    sources: IngestSources, matched: MatchedScenes, grid: AnalysisGrid, inside_aoi: np.ndarray, max_candidates: int,
) -> tuple[list[dict[str, Any]], float]:
    """The overpass with the largest **AOI** clear fraction (from SCL), or ``([], 0)``."""
    from floodfusion.data.sources import read_s2_scl
    from floodfusion.masks import quality_from_scl

    if not matched.s2_ids:
        return [], 0.0
    items = _items_by_id(sources.client, "sentinel-2-l2a", matched.s2_ids)
    passes: dict[str, list[dict[str, Any]]] = {}
    for item, when in zip(items, matched.s2_datetimes or [""] * len(items)):
        passes.setdefault(str(when)[:16] or item["id"], []).append(item)
    ranked = sorted(passes.values(), key=lambda p: np.nanmean(
        [float(i["properties"].get("eo:cloud_cover", math.nan)) for i in p]))
    best: tuple[list[dict[str, Any]], float] = ([], 0.0)
    for group in ranked[:max_candidates]:
        quality = quality_from_scl(read_s2_scl(group, grid))
        clear = float((quality[inside_aoi] >= CLEAR_QUALITY).mean()) if inside_aoi.any() else 0.0
        LOGGER.info("S2 pass %s: AOI clear fraction %.1f %%", group[0]["id"], 100 * clear)
        if clear > best[1]:
            best = (group, clear)
    return best


def _terrain(dem_items: Sequence[dict[str, Any]], grid: AnalysisGrid, theta: np.ndarray, phi: float) -> dict[str, np.ndarray]:
    from floodfusion.data.sources import read_dem
    from floodfusion.masks import (
        cast_shadow_and_layover,
        hand_on_dem_support,
        horn_slope_aspect,
        local_incidence_angle,
    )

    dem = read_dem(dem_items, grid)
    if not np.isfinite(dem).any():
        raise ValueError("DEM has no finite values over the AOI")
    dem = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    px = grid.pixel_size
    slope, aspect = horn_slope_aspect(dem, px, px)
    shadow, layover = cast_shadow_and_layover(dem, theta, phi, px)
    return {
        "dem": dem, "slope": slope, "theta_loc": local_incidence_angle(slope, aspect, theta, phi),
        "shadow": shadow, "layover": layover, "hand": hand_on_dem_support(dem, px),
    }


def _qa_coregistration(sar: Mapping[str, np.ndarray], terrain: Mapping[str, np.ndarray],
                       reflectance: np.ndarray | None) -> dict[str, Any]:
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
        "pre_vs_post": summary(median_displacement(structure_image(pre, log=False), structure_image(post, log=False))),
        "terrain_simulation_vs_post": summary(validate_geocoding(
            post, terrain["theta_loc"], shadow=terrain["shadow"], layover=terrain["layover"])),
    }
    if reflectance is not None:
        nir = np.nan_to_num(reflectance[1], nan=float(np.nanmedian(reflectance[1])))
        qa["sar_vs_optical_nir"] = summary(median_displacement(
            structure_image(post, log=False), structure_image(nir, log=True)))
    measured = [v["median_abs_displacement"] for v in qa.values() if v["n_windows"] > 0]
    qa["p1_m1_pass"] = bool(measured) and all(m < 1.0 for m in measured)
    return qa


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


def _optical_layers(sources: IngestSources, event: EventConfig, matched: MatchedScenes,
                    grid: AnalysisGrid, inside_aoi: np.ndarray) -> _OpticalLayers:
    from floodfusion.data.sources import read_s2_reflectance
    from floodfusion.masks import quality_from_scl

    items, clear = _select_optical(sources, matched, grid, inside_aoi, event.max_s2_candidates)
    if not items or clear < MIN_OPTICAL_CLEAR_FRACTION:
        LOGGER.info("%s: no usable Sentinel-2 overpass (best AOI clear fraction %.1f %%): SAR-only event",
                    event.event_id, 100 * clear)
        return _OpticalLayers(tuple(items), clear, None, None, None)
    rho, scl, offsets = read_s2_reflectance(items, grid)
    return _OpticalLayers(tuple(items), clear, rho, quality_from_scl(scl), offsets)


def _reference_layers(sources: IngestSources, event: EventConfig, grid: AnalysisGrid) -> dict[str, np.ndarray]:
    """AOI, permanent water (JRC >= 80 % OR the CEMS reference water), land cover, label."""
    from floodfusion.data.sources import read_jrc_occurrence, read_worldcover
    from floodfusion.masks import permanent_water_from_occurrence

    if event.aoi_path is None:
        LOGGER.warning("%s: no aoi_path — every unlabelled pixel of the grid is scored as dry", event.event_id)
        inside_aoi = np.ones((grid.height, grid.width), dtype=bool)
    else:
        inside_aoi = rasterise_vector(event.aoi_path, grid, layer=event.aoi_layer)
    occurrence = read_jrc_occurrence(_items_in_bbox(sources.client, "jrc-gsw", event.bbox), grid)
    permanent = permanent_water_from_occurrence(np.nan_to_num(occurrence, nan=0.0))
    if event.reference_water_path is not None:
        permanent = permanent | rasterise_vector(event.reference_water_path, grid, layer=event.reference_water_layer)
    worldcover = _items_in_bbox(sources.client, "esa-worldcover", event.bbox)
    latest = max((str(i["properties"].get("start_datetime", "")) for i in worldcover), default="")
    landcover = read_worldcover([i for i in worldcover if str(i["properties"].get("start_datetime", "")) == latest], grid)
    # Flood = observed water AND NOT permanent water (docs/05 section 3.3).
    label = rasterise_vector(event.label_path, grid, layer=event.label_layer, where=event.label_where) & ~permanent
    return {"inside_aoi": inside_aoi, "permanent": permanent, "landcover": landcover, "label": label}


def _event_meta(event: EventConfig, matched: MatchedScenes, grid: AnalysisGrid, annotations: Sequence[S1Annotation],
                phi: float, theta: np.ndarray, optical: _OpticalLayers, layers: Mapping[str, np.ndarray],
                validity: np.ndarray) -> dict[str, Any]:
    from floodfusion.utils.env import git_state

    valid = validity == 0
    return {
        "event_id": event.event_id, "epsg": grid.epsg, "transform": list(grid.transform),
        "width": grid.width, "height": grid.height, "grid_notes": list(grid.notes),
        "relative_orbit": matched.relative_orbit, "orbit_state": matched.orbit_state,
        "look_azimuth_deg": phi, "heading_deg": [a.heading_deg for a in annotations],
        "incidence_deg": [float(np.nanmin(theta)), float(np.nanmax(theta))],
        "s1_post_frames": list(matched.s1_post_frames), "s1_pre_frames": list(matched.s1_pre_frames),
        "s1_post_datetime": matched.s1_post_datetime, "s1_pre_datetime": matched.s1_pre_datetime,
        "has_optical": optical.available,
        "s2_items": [i["id"] for i in optical.items] if optical.available else [],
        "s2_aoi_clear_fraction": optical.clear_fraction, "s2_offsets": optical.offsets,
        "dem_ids": list(matched.dem_ids),
        "flood_fraction_of_valid": float((layers["label"] & valid).sum() / max(int(valid.sum()), 1)),
        "permanent_water_fraction": float(layers["permanent"].mean()),
        "shadow_fraction": float(layers["shadow"].mean()), "layover_fraction": float(layers["layover"].mean()),
        "outside_aoi_fraction": float((~layers["inside_aoi"]).mean()),
        "nodata_fraction": float(layers["nodata"].mean()),
        "event_config": {k: str(v) for k, v in vars(event).items()}, **git_state(),
    }


def ingest_event(
    event: EventConfig,
    out_root: Path | str,
    cfg: ExperimentConfig,
    *,
    sources: IngestSources | None = None,
    split_epsg: int = DEFAULT_SPLIT_EPSG,
) -> dict[str, Any]:
    """Ingest one event into ``<out_root>/events/<event_id>/``; returns its ``event.json``.

    GDAL's HTTP reads run with explicit timeouts and retries (:data:`_GDAL_ENV`).
    """
    import rasterio

    with rasterio.Env(**_GDAL_ENV):
        return _ingest_event(event, out_root, cfg, sources=sources, split_epsg=split_epsg)


def _ingest_event(
    event: EventConfig,
    out_root: Path | str,
    cfg: ExperimentConfig,
    *,
    sources: IngestSources | None,
    split_epsg: int,
) -> dict[str, Any]:
    from floodfusion.masks import build_validity_mask

    profile = profile_for(event.catalogue_url)
    if profile.name != "planetary_computer":
        raise ValueError("pixel ingest reads gamma0 from Planetary Computer sentinel-1-rtc (Route B); "
                         f"{event.catalogue_url} is usable for discovery (floodfusion.data.stac) only")
    sources = sources or IngestSources.open(event.catalogue_url)
    event_dir = Path(out_root) / "events" / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    matched = search_matched_pairs(event.query(), event.catalogue_url, client=sources.client, profile=profile)
    save_matched_scenes([matched], event_dir / "scenes.json")
    grid = build_analysis_grid(event.bbox, cfg.grid.pixel_size_m)
    if grid.width * grid.height > MAX_GRID_PIXELS:
        raise ValueError(f"{event.event_id}: a {grid.width} x {grid.height} grid exceeds {MAX_GRID_PIXELS} pixels "
                         "(~200 B/pixel in memory) — split the AOI or check the bbox")
    LOGGER.info("%s: orbit %d %s, grid EPSG:%d %dx%d", event.event_id, matched.relative_orbit,
                matched.orbit_state, grid.epsg, grid.width, grid.height)
    sar, rtc_qa = _read_sar(sources, matched, grid)
    annotations = _read_geometry(sources, matched)
    phi = _circular_mean_deg([a.look_azimuth_deg for a in annotations])
    theta = incidence_on_grid(annotations, grid)
    terrain = _terrain(_items_by_id(sources.client, profile.dem_collection, matched.dem_ids), grid, theta, phi)
    reference = _reference_layers(sources, event, grid)
    optical = _optical_layers(sources, event, matched, grid, reference["inside_aoi"])

    nodata = ~np.logical_and.reduce([np.isfinite(sar[k]) for k in ("vv_post_db", "vh_post_db", "vv_pre_db", "vh_pre_db")])
    validity = build_validity_mask(terrain["layover"], terrain["shadow"], reference["permanent"], nodata, None,
                                   slope_max_deg=None, outside_aoi=~reference["inside_aoi"])
    stack = encode_stack(
        vv_post_db=sar["vv_post_db"], vh_post_db=sar["vh_post_db"], vv_pre_db=sar["vv_pre_db"],
        vh_pre_db=sar["vh_pre_db"], reflectance=optical.reflectance, slope_deg=terrain["slope"],
        hand_m=terrain["hand"], local_incidence_deg=terrain["theta_loc"],
        label=np.where(reference["inside_aoi"], reference["label"].astype(np.float64), np.nan),
        validity=validity, optical_quality=optical.quality, landcover=reference["landcover"],
        grid_cfg=cfg.grid, norm=cfg.norm,
    )
    stack_path = write_event_stack(event_dir / "stack.tif", stack, grid, nodata=cfg.grid.nodata_int16, tags={
        "event_id": event.event_id, "s1_post": matched.s1_post_id, "s1_pre": matched.s1_pre_id,
        "s2": ",".join(i["id"] for i in optical.items) if optical.available else "",
        "look_azimuth_deg": f"{phi:.3f}",
        "generator": "floodfusion.data.ingest (Route B: Planetary Computer sentinel-1-rtc)",
    })
    basins = _event_basins(event, grid)
    records, counts = extract_chips(
        stack_path, event_id=event.event_id, basins=basins, data_cfg=cfg.data, grid_cfg=cfg.grid,
        relative_orbit=matched.relative_orbit, orbit_state=matched.orbit_state, has_optical=optical.available,
    )
    records = _to_split_crs(records, grid.epsg, split_epsg)
    _write_event_basins(event_dir / "basins.gpkg", basins, grid.epsg, split_epsg, event.event_id)
    write_manifest(records, event_dir / "chips.parquet",
                   provenance={"event_id": event.event_id, "split_epsg": split_epsg})
    meta = {
        **_event_meta(event, matched, grid, annotations, phi, theta, optical,
                      {**reference, **terrain, "nodata": nodata}, validity),
        "chip_counts": counts, "n_chips": len(records), "basins": sorted(basins), "rtc_qa": rtc_qa,
        "coregistration_qa": _qa_coregistration(sar, terrain, optical.reflectance),
    }
    atomic_write_bytes(event_dir / "event.json", json.dumps(meta, indent=2, default=str).encode("utf-8"))
    if not meta["coregistration_qa"]["p1_m1_pass"]:
        LOGGER.warning("%s: P1-M1 co-registration check did not pass: %s", event.event_id, meta["coregistration_qa"])
    return meta


def _to_split_crs(records: Sequence[ChipRecord], src_epsg: int, dst_epsg: int) -> list[ChipRecord]:
    from pyproj import Transformer

    if src_epsg == dst_epsg or not records:
        return list(records)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    xs, ys = tr.transform([r.centroid_x for r in records], [r.centroid_y for r in records])
    return [replace(r, centroid_x=float(x), centroid_y=float(y)) for r, x, y in zip(records, xs, ys)]


def _write_event_basins(path: Path, basins: Mapping[str, Any], src_epsg: int, dst_epsg: int, event_id: str) -> None:
    import geopandas as gpd

    frame = gpd.GeoDataFrame({"basin_id": list(basins), "event_id": [event_id] * len(basins)},
                             geometry=list(basins.values()), crs=f"EPSG:{src_epsg}").to_crs(epsg=dst_epsg)
    tmp = path.with_name(f".tmp.{path.name}")
    frame.to_file(tmp, driver="GPKG", layer="basins")
    tmp.replace(path)


def build_dataset(out_root: Path | str, cfg: ExperimentConfig, *, split_epsg: int = DEFAULT_SPLIT_EPSG) -> str:
    """Merge every ingested event into ``chips/manifest.parquet`` and ``basins.gpkg``.

    Returns the manifest hash. Basins shared by several events (same HydroBASINS id) are
    kept once: they are one split group.
    """
    import geopandas as gpd
    import pandas as pd

    from floodfusion.utils.env import git_state

    root = Path(out_root)
    event_dirs = sorted(p for p in (root / "events").glob("*") if (p / "chips.parquet").is_file())
    if not event_dirs:
        raise FileNotFoundError(f"no ingested events under {root / 'events'}")
    records: list[ChipRecord] = []
    frames = []
    for d in event_dirs:
        records.extend(load_manifest(d / "chips.parquet"))
        frame = gpd.read_file(d / "basins.gpkg", layer="basins")
        if frame.crs is None or frame.crs.to_epsg() != split_epsg:
            raise ValueError(f"{d}: basins are not in EPSG:{split_epsg}; re-ingest with one --split-epsg")
        frames.append(frame)
    basins = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{split_epsg}")
    basins = basins.drop_duplicates(subset="basin_id", keep="first")
    tmp = root / ".tmp.basins.gpkg"
    basins.to_file(tmp, driver="GPKG", layer="basins")
    tmp.replace(root / "basins.gpkg")
    return write_manifest(records, root / Path(cfg.data.chip_root).name / cfg.data.manifest_name, provenance={
        "generator": "floodfusion.data.ingest", "events": [d.name for d in event_dirs],
        "split_epsg": split_epsg, "route": "B (Planetary Computer sentinel-1-rtc)", **git_state(),
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest flood events from public archives (Planetary Computer).")
    parser.add_argument("--event", type=Path, action="append", required=True, help="event YAML (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--split-epsg", type=int, default=DEFAULT_SPLIT_EPSG)
    parser.add_argument("--force", action="store_true", help="re-ingest events that already have a stack")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, overlays=args.overlay)
    failures = 0
    for path in args.event:
        event = load_event_config(path)
        if (args.out / "events" / event.event_id / "chips.parquet").is_file() and not args.force:
            LOGGER.info("%s already ingested (use --force to redo)", event.event_id)
            continue
        try:
            meta = ingest_event(event, args.out, cfg, split_epsg=args.split_epsg)
            LOGGER.info("%s: %d chips, flood %.1f %% of valid pixels", event.event_id, meta["n_chips"],
                        100 * meta["flood_fraction_of_valid"])
        except LookupError as exc:
            # docs/05 2.1 criterion 3: an event without a same-orbit pre scene is DROPPED.
            LOGGER.error("%s dropped: %s", event.event_id, exc)
            failures += 1
        except Exception as exc:  # noqa: BLE001 - one event's failure must not abort the batch
            # Error texts of HTTP/GDAL failures embed signed URLs: never print the token.
            LOGGER.error("%s failed: %s: %s", event.event_id, type(exc).__name__, redact(str(exc)))
            failures += 1
    digest = build_dataset(args.out, cfg, split_epsg=args.split_epsg)
    LOGGER.info("manifest %s", digest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
