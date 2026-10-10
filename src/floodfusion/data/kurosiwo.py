r"""Kuro Siwo (Bountos et al., NeurIPS 2024): change-aware flood labels drawn on radar, re-ingested.

Kuro Siwo annotates 43 flood events (2015-2022) on Sentinel-1 at 10 m with three classes:
*flood* (water the event brought), *permanent water* (its ``hydro`` layer) and no water. Its
labels are the opposite of Sen1Floods11's in two ways that matter here: they were drawn on
**radar**, and they separate the flood from water that was there before. Only the annotation
**polygons** are used (``Orion-AI-Lab/KuroSiwo-annotations`` at a pinned commit, sha256
verified); every input is fetched again from Planetary Computer through the assembly
Sen1Floods11 uses (:mod:`floodfusion.data.sen1floods11`).

**Dates.** The polygons of most events carry no acquisition date. The post-event Sentinel-1
date of every event comes from the dataset's own GeoTIFF names (``MS1_*_<act>_<aoi>_<date>.tif``
in ``orion-ai-lab/Kuro-Siwo-GeoTIFFs``) and is committed in ``configs/datasets/kurosiwo_events.yaml``;
an event without one is excluded.

**Acquisition.** On that date, the Sentinel-1 acquisition covering the largest share of the AOI
(ties: the earlier) — the image the annotators most plausibly drew on.

**Chips.** 5.12 km squares on a UTM grid inside AOI ∩ acquisition footprint (at least
:data:`MIN_TILE_INSIDE` inside). Per event, with a generator seeded by the event's name, up to
:data:`N_FLOOD_TILES` with flood >= :data:`FLOOD_TILE_FRACTION` and up to :data:`N_DRY_TILES` below.

**Labels.** Two definitions on identical chips: ``flood`` (the dataset's own: 1 on the event
layer, 0 elsewhere in the AOI) and ``water`` (:func:`relabel_water`: 1 on event ∪ hydro, the
Sen1Floods11-like "water on a date"). In both, JRC permanent water (occurrence >= 80 %) is
removed exactly as for Sen1Floods11. Only the label differs between the two chip sets.

**Geometry.** Labels drawn on radar should sit on the SAR grid; each acquisition's label offset
(label edges vs SAR) and optical offset (SWIR vs SAR) are measured over its chips and applied
only when they are measurements (Sen1Floods11's rule: >= 12 chips, 95 % interval <= 1 px).

**Groups.** Events whose AOIs lie within :data:`GROUP_DISTANCE_KM` of each other form one split
group (single linkage): a region flooded twice must not train one model and test it.

**Licence.** Imagery CC-BY-4.0 (``orion-ai-lab/Kuro-Siwo-Webdataset``); annotation repository
MIT. Nothing derived from it is redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.archive import GDAL_ENV, PC_STAC_URL, IngestSources, items_by_id
from floodfusion.data.assembly import (
    SHIFT_MAX_CI_PX,
    GridReference,
    GridTask,
    assemble_on_grid,
    expand_grid,
)
from floodfusion.data.chipping import AnalysisGrid, build_analysis_grid
from floodfusion.data.http import redacted_traceback
from floodfusion.data.passes import Acquisition, acquisitions, best_covering
from floodfusion.data.records import atomic_write_bytes
from floodfusion.data.stac import PROFILES, EventQuery, NoMatchingPairError, redact

__all__ = [
    "ANNOTATIONS_COMMIT",
    "ANNOTATIONS_SHA256",
    "KsAoi",
    "Tile",
    "candidate_tiles",
    "choose_shift",
    "download",
    "event_groups",
    "ingest_kurosiwo",
    "list_aois",
    "load_event_dates",
    "main",
    "relabel_water",
    "sample_tiles",
    "shift_sample",
]

LOGGER = logging.getLogger(__name__)

ANNOTATIONS_COMMIT = "b930dd278dfec69986f4e33c742958c1199ec86c"
ANNOTATIONS_SHA256 = "0a4b33eadb74a6469ffd621ce9c5dc0bfae67e49721d5f6203fcbe9694cae59b"
_HOST = "codeload.github.com"
_URL = f"https://{_HOST}/Orion-AI-Lab/KuroSiwo-annotations/tar.gz/{ANNOTATIONS_COMMIT}"
_MAX_TARBALL_BYTES = 400 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
#: ``<repo>-<sha>/polygons/<event>/aoi/<nn>/<version>/{aoi,event,hydro}/<name>.<shapefile part>``
_MEMBER = re.compile(rf"KuroSiwo-annotations-{ANNOTATIONS_COMMIT}/polygons/(EMSR\d+_[\w-]+)/aoi/(\d{{2}})/"
                     r"(r\d+_v\d+)/(aoi|event|hydro)/[\w-]+\.(shp|shx|dbf|prj|cpg)", re.ASCII)
_MAX_MEMBERS = 5000

#: Chip side (px at 10 m): Sen1Floods11's 512 x 512 chips.
CHIP_PX = 512
MIN_TILE_INSIDE = 0.9
FLOOD_TILE_FRACTION = 0.01
N_FLOOD_TILES = 12
N_DRY_TILES = 4
GROUP_DISTANCE_KM = 100.0
#: An acquisition must cover at least this share of an AOI to be used for it.
MIN_AOI_COVERAGE = 0.1
#: Sentinel-2 is searched this many days either side of the post-event Sentinel-1 date.
OPTICAL_WINDOW_DAYS = 2
#: 256 px = 2.56 km of terrain context around each chip (as Sen1Floods11).
DEFAULT_MARGIN_PX = 256
#: One metric CRS for every chip centroid (the events span six continents).
SPLIT_EPSG = 3857
GROUP_PREFIX = "KS_"
_MIN_SHIFT_GRIDS = 12
#: Chips read per AOI to measure its label and optical offsets: the training sample plus a seeded
#: draw from the other candidate tiles, up to this many (more chips, more acquisitions measured).
SHIFT_SAMPLE_TILES = 48


# ---------------------------------------------------------------------------
# Download: one pinned tarball, sha256-verified, shapefile parts only
# ---------------------------------------------------------------------------
def _http_get(url: str) -> bytes:
    from floodfusion.data.http import capped_get, check_https_url

    return capped_get(url, check=lambda u: check_https_url(u, hosts=(_HOST,)), cap=_MAX_TARBALL_BYTES,
                      timeout_s=600, retries=3)


def download(raw_root: Path | str, *, get: Callable[[str], bytes] = _http_get) -> int:
    """Fetch the pinned annotation tarball and extract its shapefile layers; return files extracted.

    Resumable: ``raw_root/polygons/.complete`` marks a finished extraction. A tarball whose
    sha256 differs from :data:`ANNOTATIONS_SHA256` is refused before anything is written.
    """
    root = Path(raw_root)
    done = root / "polygons" / ".complete"
    if done.is_file():
        return 0
    data = get(_URL)
    digest = hashlib.sha256(data).hexdigest()
    if digest != ANNOTATIONS_SHA256:
        raise OSError(f"Kuro Siwo annotations: sha256 {digest} != pinned {ANNOTATIONS_SHA256}")
    n, total = 0, 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > _MAX_MEMBERS:
            raise OSError(f"Kuro Siwo annotations: {len(members)} members (cap {_MAX_MEMBERS})")
        for member in members:
            match = _MEMBER.fullmatch(member.name)
            if not member.isfile() or match is None:
                continue
            if member.size > _MAX_MEMBER_BYTES or total + member.size > _MAX_EXTRACTED_BYTES:
                raise OSError(f"{member.name}: over the extraction caps")
            event, aoi, version, layer, _ = match.groups()
            dest = root / "polygons" / event / aoi / version / layer / Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(dest, source.read(member.size + 1)[:member.size])
            total += member.size
            n += 1
    if not n:
        raise OSError("Kuro Siwo annotations: no shapefile member matched the expected layout")
    atomic_write_bytes(done, json.dumps({"commit": ANNOTATIONS_COMMIT, "sha256": digest, "files": n}).encode("utf-8"))
    LOGGER.info("Kuro Siwo annotations: %d files extracted", n)
    return n


# ---------------------------------------------------------------------------
# Events, AOIs, dates, groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KsAoi:
    event: str          # "EMSR497_Germany"
    aoi: str            # "01"
    layers: Path        # .../polygons/<event>/<aoi>/<version>

    @property
    def stack_prefix(self) -> str:
        return f"KS_{self.event}_{self.aoi}"


def _version_key(name: str) -> tuple[int, int]:
    r, v = re.fullmatch(r"r(\d+)_v(\d+)", name).groups()  # type: ignore[union-attr]
    return int(r), int(v)


def list_aois(raw_root: Path | str) -> list[KsAoi]:
    """Every AOI with flood and AOI layers, at its latest annotation version."""
    out = []
    for aoi_dir in sorted((Path(raw_root) / "polygons").glob("EMSR*/[0-9][0-9]")):
        versions = sorted((d for d in aoi_dir.iterdir() if re.fullmatch(r"r\d+_v\d+", d.name)),
                          key=lambda d: _version_key(d.name))
        if versions and (versions[-1] / "event").is_dir() and (versions[-1] / "aoi").is_dir():
            out.append(KsAoi(aoi_dir.parent.name, aoi_dir.name, versions[-1]))
    return out


def _layer(aoi: KsAoi, name: str) -> Any:
    """One layer (``aoi``, ``event``, ``hydro``) as a GeoDataFrame in EPSG:4326 (empty if absent)."""
    import geopandas as gpd
    import pandas as pd

    files = sorted((aoi.layers / name).glob("*.shp"))
    frames = [gpd.read_file(f) for f in files]
    frames = [f.set_crs(4326) if f.crs is None else f.to_crs(4326) for f in frames if len(f)]
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    frame = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    if name == "aoi":   # a handful of vertices; the flood layers are rasterised, never clipped, as drawn
        frame["geometry"] = frame.geometry.make_valid()
    return frame


def load_event_dates(path: Path | str) -> dict[str, tuple[date, ...]]:
    """``{event folder: post-event dates}`` from ``configs/datasets/kurosiwo_events.yaml``."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for event, spec in (data.get("events") or {}).items():
        days = tuple(sorted(date.fromisoformat(str(d)) for d in (spec.get("post") or [])))
        if days:
            out[str(event)] = days
    return out


def event_groups(geometries: Mapping[str, Any], *, distance_km: float = GROUP_DISTANCE_KM) -> dict[str, str]:
    """``{event: group}``: single-linkage clusters of events whose AOIs lie within ``distance_km``.

    A group is named after its alphabetically first event, so the names do not depend on order.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    names = sorted(geometries)
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = geometries[a], geometries[b]
            if ga.intersects(gb):
                close = True
            else:
                from shapely.ops import nearest_points

                pa, pb = nearest_points(ga, gb)
                close = geod.inv(pa.x, pa.y, pb.x, pb.y)[2] / 1000.0 <= distance_km
            if close:
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    return {n: f"{GROUP_PREFIX}{find(n)}" for n in names}


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tile:
    aoi: str                                   # KsAoi.stack_prefix
    bbox: tuple[float, float, float, float]    # EPSG:4326
    flood_fraction: float
    utm: tuple[int, float, float]              # epsg, x0, y0 (south-west corner, metres)

    @property
    def tile_id(self) -> str:
        return f"{self.aoi}_{int(self.utm[1]):d}_{int(self.utm[2]):d}"


#: Resolution (m) of the raster the tile planner measures flood shares on: 64 px per 5.12 km tile.
_PLAN_RES_M = 80.0


def candidate_tiles(region: Any, flood: Any, *, aoi: str, size_m: float = CHIP_PX * 10.0,
                    min_inside: float = MIN_TILE_INSIDE) -> list[Tile]:
    """Squares of ``size_m`` on a UTM grid (origin snapped to ``size_m``) mostly inside ``region``.

    ``flood`` is one geometry or a sequence of them. Its share of each square is measured on an
    80 m raster: the event layers are rasterised as drawn, never dissolved or repaired (repairing
    Kuro Siwo's 1 381 invalid Pakistan polygons takes longer than the whole ingest).
    """
    import shapely
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from floodfusion.data.chipping import utm_epsg_for

    if region.is_empty:
        return []
    epsg = utm_epsg_for(region.centroid.x, region.centroid.y)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)

    def project(geometry: Any) -> Any:
        return shapely.transform(geometry, lambda c: np.column_stack(to_utm.transform(c[:, 0], c[:, 1])))

    region_m = project(region)
    parts = flood if isinstance(flood, Sequence) else ([] if flood is None or flood.is_empty
                                                       else list(getattr(flood, "geoms", [flood])))
    x_min, y_min, x_max, y_max = region_m.bounds
    x0, y1 = np.floor(x_min / size_m) * size_m, np.ceil(y_max / size_m) * size_m
    xs = np.arange(x0, x_max, size_m)
    ys = np.arange(np.floor(y_min / size_m) * size_m, y_max, size_m)
    if not len(xs) or not len(ys):
        return []
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys, indexing="ij"))
    squares = shapely.box(gx, gy, gx + size_m, gy + size_m)
    shapely.prepare(region_m)
    inside = shapely.contains(region_m, squares)
    edge = ~inside & shapely.intersects(region_m, squares)
    share = inside.astype(np.float64)
    if edge.any():
        share[edge] = shapely.area(shapely.intersection(region_m, squares[edge])) / (size_m * size_m)
    keep = np.flatnonzero(share >= min_inside)
    wet = np.zeros(len(squares))
    geoms = [project(g) for g in parts if g is not None and not g.is_empty]
    if geoms and len(keep):
        px = int(round(size_m / _PLAN_RES_M))
        width, height = int(round((xs[-1] + size_m - x0) / _PLAN_RES_M)), int(round((y1 - ys[0]) / _PLAN_RES_M))
        raster = rasterize(((g, 1) for g in geoms), out_shape=(height, width),
                           transform=from_origin(x0, y1, _PLAN_RES_M, _PLAN_RES_M), fill=0, dtype="uint8")
        for i in keep:
            c, r = int(round((gx[i] - x0) / _PLAN_RES_M)), int(round((y1 - gy[i] - size_m) / _PLAN_RES_M))
            wet[i] = float(raster[r:r + px, c:c + px].mean())
    out = []
    for i in keep:
        sx, sy = float(gx[i]), float(gy[i])
        lon, lat = to_ll.transform([sx, sx + size_m, sx, sx + size_m], [sy, sy, sy + size_m, sy + size_m])
        out.append(Tile(aoi, (min(lon), min(lat), max(lon), max(lat)), wet[i], (int(epsg), sx, sy)))
    return out


def sample_tiles(tiles: Sequence[Tile], event: str, *, n_flood: int = N_FLOOD_TILES,
                 n_dry: int = N_DRY_TILES) -> list[Tile]:
    """Up to ``n_flood`` flooded and ``n_dry`` dry tiles, drawn by a generator seeded with ``event``."""
    seed = int(hashlib.sha256(event.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    ordered = sorted(tiles, key=lambda t: t.tile_id)
    wet = [t for t in ordered if t.flood_fraction >= FLOOD_TILE_FRACTION]
    dry = [t for t in ordered if t.flood_fraction < FLOOD_TILE_FRACTION]
    pick = [wet[i] for i in sorted(rng.permutation(len(wet))[:n_flood])]
    pick += [dry[i] for i in sorted(rng.permutation(len(dry))[:n_dry])]
    return pick


# ---------------------------------------------------------------------------
# Reference layers and shifts
# ---------------------------------------------------------------------------
def _rasterise(geometries: Any, grid: AnalysisGrid) -> np.ndarray:
    """Polygons -> mask on the grid (pixel centres inside). Invalid polygons rasterise as drawn.

    Only polygons whose bounding box touches the grid are projected: an event layer can hold
    tens of thousands (Kuro Siwo's Pakistan: 47 677), and a chip is 5 km.
    """
    from rasterio.features import rasterize

    from floodfusion.data.seasonal import _grid_polygon

    if len(geometries):
        geometries = geometries.iloc[geometries.sindex.query(_grid_polygon(grid))]
    frame = geometries.to_crs(grid.epsg)
    shapes = [(g, 1) for g in frame.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((grid.height, grid.width), dtype=bool)
    return rasterize(shapes, out_shape=(grid.height, grid.width), transform=grid.affine, fill=0,
                     all_touched=False, dtype="uint8").astype(bool)


def label_on_grid(aoi_frame: Any, flood_frame: Any, grid: AnalysisGrid) -> np.ndarray:
    """The ``flood`` label: 1 flood, 0 elsewhere in the AOI, NaN outside it."""
    inside = _rasterise(aoi_frame, grid)
    return np.where(inside, _rasterise(flood_frame, grid).astype(np.float64), np.nan)


def chip_reference(aoi_frame: Any, flood_frame: Any, grid: AnalysisGrid, jrc_items: Sequence[Mapping[str, Any]],
                   worldcover_items: Sequence[Mapping[str, Any]]) -> GridReference:
    """Kuro Siwo's flood label on ``grid``, minus JRC permanent water (as Sen1Floods11)."""
    from floodfusion.data.sources import read_jrc_occurrence, read_worldcover
    from floodfusion.masks import permanent_water_from_occurrence

    label = label_on_grid(aoi_frame, flood_frame, grid)
    if jrc_items:
        permanent = permanent_water_from_occurrence(np.nan_to_num(read_jrc_occurrence(jrc_items, grid), nan=0.0))
    else:
        permanent = np.zeros((grid.height, grid.width), dtype=bool)
    label = np.where(permanent & (label == 1), 0.0, label)
    landcover = (read_worldcover(worldcover_items, grid) if worldcover_items
                 else np.zeros((grid.height, grid.width), dtype=np.uint8))
    return GridReference(label=label, inside_aoi=np.isfinite(label), permanent=permanent, landcover=landcover)


def choose_shift(estimates: Mapping[str, Mapping[str, Any]], *, max_ci_px: float = SHIFT_MAX_CI_PX,
                 min_grids: int = _MIN_SHIFT_GRIDS) -> dict[str, Any]:
    """Label and optical shifts for labels drawn on **radar**: each from its own offset to the SAR.

    The label moves by its edge offset to the SAR and the optical by its SWIR offset to the SAR,
    each only when that offset is a measurement (>= ``min_grids`` chips, 95 % interval within
    ``max_ci_px``); otherwise it stays.
    """
    def measured(name: str) -> bool:
        e = estimates.get(name) or {}
        return e.get("n_grids", 0) >= min_grids and float(e.get("ci_halfwidth_px", float("nan"))) <= max_ci_px

    def shift(name: str) -> list[int]:
        return [int(v) for v in estimates[name]["shift_px"]] if measured(name) else [0, 0]

    return {"label_shift_px": shift("label"), "label_source": "label" if measured("label") else "none",
            "optical_shift_px": shift("optical"), "optical_source": "optical" if measured("optical") else "none",
            "estimates": {k: dict(v) for k, v in estimates.items()}}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _query(aoi: KsAoi, acq: Acquisition, bbox: Sequence[float]) -> EventQuery:
    day = datetime.combine(acq.when.date(), datetime.min.time())
    window = timedelta(days=OPTICAL_WINDOW_DAYS)
    return EventQuery(aoi.stack_prefix, tuple(bbox), day, day, pre_event_window_days=60,
                      relative_orbit=acq.relative_orbit, orbit_state=acq.orbit_state,
                      optical_start=day - window, optical_end=day + window)


def _shift_windows(tile: Tile, archive: Any, aoi_frame: Any, flood_frame: Any,
                   cfg: ExperimentConfig) -> dict[str, np.ndarray] | None:
    import rasterio

    from floodfusion.data.chipping import CLEAR_QUALITY
    from floodfusion.data.sen1floods11 import _restrict, chip_shift_windows
    from floodfusion.data.sources import read_s1_rtc, read_s2_reflectance
    from floodfusion.masks import quality_from_scl

    try:
        matched = _restrict(archive.matched, archive.by_id, tile.bbox)
    except NoMatchingPairError:
        return None
    grid = build_analysis_grid(tile.bbox, cfg.grid.pixel_size_m)
    with rasterio.Env(**GDAL_ENV):
        vv, _ = read_s1_rtc(items_by_id(archive.catalogue, PROFILES["planetary_computer"].s1_collection,
                                        matched.s1_post_frames), grid, "VV")
        rho = None
        if matched.s2_ids:
            rho, scl, _ = read_s2_reflectance(items_by_id(archive.catalogue, "sentinel-2-l2a", matched.s2_ids), grid)
            rho = np.where(quality_from_scl(scl)[None] >= CLEAR_QUALITY, rho, np.nan)
    return chip_shift_windows(vv, label_on_grid(aoi_frame, flood_frame, grid), rho)


def _measure_shift(tiles: Sequence[Tile], archive: Any, aoi_frame: Any, flood_frame: Any,
                   cfg: ExperimentConfig, workers: int) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    from floodfusion.data.assembly import pooled_shift
    from floodfusion.data.sen1floods11 import _json_ready

    def read(tile: Tile) -> dict[str, np.ndarray] | None:
        try:
            return _shift_windows(tile, archive, aoi_frame, flood_frame, cfg)
        except Exception as exc:  # noqa: BLE001 - one unreadable chip must not decide the shift
            LOGGER.warning("%s: shift inputs unreadable, skipped: %s", tile.tile_id, redact(str(exc))[:200])
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        per_chip = [r for r in pool.map(read, tiles) if r is not None]
    estimates = {k: pooled_shift([c[k] for c in per_chip if k in c]) for k in ("optical", "label", "label_vs_optical")}
    return _json_ready({**choose_shift(estimates), "n_chips_read": len(per_chip)})


def _ingest_aoi(aoi: KsAoi, acq: Acquisition, tiles: Sequence[Tile], group: str, out_root: Path,
                cfg: ExperimentConfig, sources: IngestSources, margin_px: int, workers: int,
                shift_tiles: Sequence[Tile] = ()) -> dict[str, str]:
    from concurrent.futures import ThreadPoolExecutor

    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from shapely.ops import unary_union

    from floodfusion.data.sen1floods11 import (
        _overlaps,
        _restrict,
        _to_grid_crs,
        _union_bbox,
        archive_for_query,
    )

    status: dict[str, Any] = {}
    todo = [t for t in tiles if not (out_root / "events" / t.tile_id / "event.json").is_file()]
    status.update({t.tile_id: "done-before" for t in tiles if t not in todo})
    if not todo:
        return status
    aoi_frame, flood_frame = _layer(aoi, "aoi"), _layer(aoi, "event")
    measure_on = tuple(shift_tiles) or tuple(tiles)
    bbox = _union_bbox([t.bbox for t in measure_on])
    try:
        archive = archive_for_query(_query(aoi, acq, bbox), bbox, sources, margin_px * cfg.grid.pixel_size_m)
        shift = _measure_shift(measure_on, archive, aoi_frame, flood_frame, cfg, workers)
    except NoMatchingPairError as exc:
        return {**status, **{t.tile_id: f"dropped: {exc}" for t in todo}}
    except Exception as exc:  # noqa: BLE001 - one AOI's failure must not abort the dataset
        LOGGER.error("%s failed\n%s", aoi.stack_prefix, redacted_traceback(exc))
        return {**status, **{t.tile_id: f"failed: {type(exc).__name__}: {redact(str(exc))[:300]}" for t in todo}}
    status["event_shift"] = shift
    polygon = unary_union([box(*t.bbox) for t in tiles])

    def run(tile: Tile) -> tuple[str, str]:
        try:
            with rasterio.Env(**GDAL_ENV):
                grid = build_analysis_grid(tile.bbox, cfg.grid.pixel_size_m)
                wide = transform_bounds(f"EPSG:{grid.epsg}", "EPSG:4326", *expand_grid(grid, margin_px).bounds)
                task = GridTask(
                    event_id=tile.tile_id, out_dir=out_root / "events" / tile.tile_id, grid=grid,
                    matched=_restrict(archive.matched, archive.by_id, tile.bbox), annotations=archive.annotations,
                    reference=chip_reference(aoi_frame, flood_frame, grid,
                                             [i for i in archive.jrc if _overlaps(i["bbox"], tile.bbox)],
                                             [i for i in archive.worldcover if _overlaps(i["bbox"], tile.bbox)]),
                    basins={group: _to_grid_crs(polygon, grid.epsg)},
                    dem_items=tuple(i for i in archive.dem if _overlaps(i["bbox"], wide)),
                    extra_meta={"kurosiwo": {"event": aoi.event, "aoi": aoi.aoi, "group": group,
                                             "post_acquisition": acq.when.isoformat(),
                                             "flood_fraction": tile.flood_fraction, "annotation_commit": ANNOTATIONS_COMMIT},
                                "event_shift": dict(shift)},
                    terrain_margin_px=margin_px, label_follows_optical=False,
                    optical_shift_px=(int(shift["optical_shift_px"][0]), int(shift["optical_shift_px"][1])),
                    label_shift_px=(int(shift["label_shift_px"][0]), int(shift["label_shift_px"][1])),
                    generator="floodfusion.data.kurosiwo (labels: Kuro Siwo polygons; inputs: Planetary Computer)",
                )
                meta = assemble_on_grid(task, cfg, IngestSources(client=archive.catalogue,
                                                                 fetch_bytes=sources.fetch_bytes),
                                        split_epsg=SPLIT_EPSG)
            if not meta["n_chips"]:
                return tile.tile_id, f"dropped: no window passed the filters {meta['chip_counts']}"
            return tile.tile_id, f"ok: {meta['n_chips']} windows, optical={meta['has_optical']}"
        except NoMatchingPairError as exc:
            return tile.tile_id, f"dropped: {exc}"
        except Exception as exc:  # noqa: BLE001 - one chip's failure must not abort the AOI
            LOGGER.error("%s failed\n%s", tile.tile_id, redacted_traceback(exc))
            return tile.tile_id, f"failed: {type(exc).__name__}: {redact(str(exc))}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for tile_id, outcome in pool.map(run, todo):
            status[tile_id] = outcome
            LOGGER.info("%s %s", tile_id, outcome)
    return status


@dataclass(frozen=True)
class _Plan:
    aoi: KsAoi
    acquisition: Acquisition
    coverage: float
    tiles: tuple[Tile, ...]          # ingested
    shift_tiles: tuple[Tile, ...]    # read only to measure the offsets (a superset of ``tiles``)


def shift_sample(tiles: Sequence[Tile], candidates: Sequence[Tile], key: str, *,
                 n: int = SHIFT_SAMPLE_TILES) -> tuple[Tile, ...]:
    """``tiles`` plus a seeded draw from the other ``candidates``, at most ``n`` in all."""
    chosen = {t.tile_id for t in tiles}
    rest = sorted((t for t in candidates if t.tile_id not in chosen), key=lambda t: t.tile_id)
    rng = np.random.default_rng(int(hashlib.sha256(f"shift:{key}".encode()).hexdigest()[:8], 16))
    extra = [rest[i] for i in sorted(rng.permutation(len(rest))[:max(0, n - len(tiles))])]
    return tuple(sorted([*tiles, *extra], key=lambda t: t.tile_id))


def plan_event(event: str, aois: Sequence[KsAoi], dates: Sequence[date], client: Any) -> list[_Plan]:
    """Per AOI the acquisition on a post date covering it best, and the event's sampled tiles."""
    plans: list[tuple[KsAoi, Acquisition, float, list[Tile]]] = []
    for aoi in aois:
        region = _layer(aoi, "aoi").geometry.union_all()
        candidates = [a for day in dates for a in acquisitions(
            client, region, datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            datetime.combine(day, datetime.max.time(), tzinfo=UTC))]
        chosen = best_covering(candidates, region)
        if chosen is None or chosen[1] < MIN_AOI_COVERAGE:
            LOGGER.warning("%s: no acquisition on %s covers >= %.0f %% of the AOI", aoi.stack_prefix,
                           [d.isoformat() for d in dates], 100 * MIN_AOI_COVERAGE)
            continue
        acq, share = chosen
        flood = list(_layer(aoi, "event").geometry)
        tiles = candidate_tiles(region.intersection(acq.footprint), flood, aoi=aoi.stack_prefix)
        plans.append((aoi, acq, share, tiles))
    picked = {t.tile_id for t in sample_tiles([t for *_, ts in plans for t in ts], event)}
    out = []
    for aoi, acq, share, candidates in plans:
        tiles = tuple(t for t in candidates if t.tile_id in picked)
        if tiles:
            out.append(_Plan(aoi, acq, share, tiles, shift_sample(tiles, candidates, aoi.stack_prefix)))
    return out


def ingest_kurosiwo(raw_root: Path | str, out_root: Path | str, cfg: ExperimentConfig, events_file: Path | str, *,
                    events: Sequence[str] | None = None, sources: IngestSources | None = None,
                    margin_px: int = DEFAULT_MARGIN_PX, workers: int = 4) -> dict[str, Any]:
    """Plan and ingest every dated event (or those of ``events``); write ``ingest_report.json``. Resumable."""
    raw_root, out_root = Path(raw_root), Path(out_root)
    dates = load_event_dates(events_file)
    aois = list_aois(raw_root)
    by_event: dict[str, list[KsAoi]] = {}
    for a in aois:
        by_event.setdefault(a.event, []).append(a)
    groups = event_groups({e: _union_geometry(a) for e, a in by_event.items()})
    names = sorted(by_event if events is None else set(events))
    sources = sources or IngestSources.open(PC_STAC_URL)
    report: dict[str, Any] = {"events": {}, "groups": groups, "excluded": {}}
    for name in names:
        if name not in dates:
            report["excluded"][name] = "no post-event date"
            continue
        plans = plan_event(name, by_event[name], dates[name], sources.client)
        if not plans:
            report["excluded"][name] = "no covering acquisition or no tile inside the AOI"
            continue
        status: dict[str, Any] = {}
        for plan in plans:
            LOGGER.info("%s AOI %s: %s o%03d (%.0f %% of the AOI), %d tiles", name, plan.aoi.aoi,
                        plan.acquisition.when.isoformat(), plan.acquisition.relative_orbit, 100 * plan.coverage,
                        len(plan.tiles))
            status[plan.aoi.aoi] = _ingest_aoi(plan.aoi, plan.acquisition, plan.tiles, groups[name], out_root, cfg,
                                               sources, margin_px, workers, shift_tiles=plan.shift_tiles)
        report["events"][name] = status
    outcomes = [o for ev in report["events"].values() for st in ev.values() for k, o in st.items()
                if k != "event_shift" and isinstance(o, str)]
    report["summary"] = {k: sum(o.startswith(k) for o in outcomes) for k in ("ok", "done-before", "dropped", "failed")}
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_root / "ingest_report.json", json.dumps(report, indent=2, default=str).encode("utf-8"))
    return report


def _union_geometry(aois: Sequence[KsAoi]) -> Any:
    from shapely.ops import unary_union

    return unary_union([_layer(a, "aoi").geometry.union_all() for a in aois])


# ---------------------------------------------------------------------------
# The "water" label: same chips, hydro added to the flood
# ---------------------------------------------------------------------------
def relabel_water(src_root: Path | str, dst_root: Path | str, raw_root: Path | str, cfg: ExperimentConfig) -> int:
    """Copy every ingested chip stack to ``dst_root`` with label = flood ∪ hydro; return stacks written.

    Windows, validity and every input band are unchanged; only the label band and each window's
    ``positive_fraction`` differ. JRC permanent water keeps label 0 and stays invalid, as in the
    ``flood`` set.
    """
    import shutil

    import rasterio

    from floodfusion.config import ValidityFlag
    from floodfusion.data.chipping import read_event_stack
    from floodfusion.data.encoding import ChipBand
    from floodfusion.data.records import load_manifest, write_manifest

    src_root, dst_root = Path(src_root), Path(dst_root)
    aois = {a.stack_prefix: a for a in list_aois(raw_root)}
    nodata = int(cfg.grid.nodata_int16)
    n = 0
    # A stack whose windows all failed the filters has no chips.parquet: nothing to relabel.
    for event_dir in sorted(p for p in (src_root / "events").glob("*")
                            if (p / "event.json").is_file() and (p / "chips.parquet").is_file()):
        out_dir = dst_root / "events" / event_dir.name
        if (out_dir / "event.json").is_file():
            continue
        meta = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
        ks = meta["kurosiwo"]
        aoi = aois[f"KS_{ks['event']}_{ks['aoi']}"]
        raw, grid, _ = read_event_stack(event_dir / "stack.tif")
        hydro = _rasterise(_layer(aoi, "hydro"), grid)
        label, validity = raw[ChipBand.LABEL], raw[ChipBand.VALIDITY].astype(np.int64)
        permanent = (validity & int(ValidityFlag.PERMANENT_WATER)) != 0
        label = np.where(hydro & (label == 0) & ~permanent, 1, label).astype(np.int16)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("basins.gpkg",):
            shutil.copy2(event_dir / name, out_dir / name)
        shutil.copy2(event_dir / "stack.tif", out_dir / "stack.tif")
        with rasterio.open(out_dir / "stack.tif", "r+") as dst:
            dst.write(label, int(ChipBand.LABEL) + 1)
        records = []
        vv = raw[ChipBand.VV_POST]
        for r in load_manifest(event_dir / "chips.parquet"):
            col, row, w, h = r.window
            sl = (slice(row, row + h), slice(col, col + w))
            valid = (validity[sl] == 0) & (label[sl] != nodata) & (vv[sl] != nodata)
            positive = valid & (label[sl] == 1)
            records.append(replace(r, raster_path=out_dir / "stack.tif",
                                   positive_fraction=float(positive.sum() / max(int(valid.sum()), 1))))
        write_manifest(records, out_dir / "chips.parquet",
                       provenance={"generator": "floodfusion.data.kurosiwo.relabel_water", "source": str(event_dir)})
        meta["kurosiwo"] = {**ks, "label_definition": "water (flood ∪ hydro)"}
        atomic_write_bytes(out_dir / "event.json", json.dumps(meta, indent=2).encode("utf-8"))
        n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kuro Siwo labels, re-ingested through floodfusion.")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch the pinned annotation tarball (sha256-verified)")
    dl.add_argument("--raw", type=Path, default=Path("data/kurosiwo_raw"))
    ing = sub.add_parser("ingest", help="sample chips and re-fetch every input from Planetary Computer")
    ing.add_argument("--raw", type=Path, default=Path("data/kurosiwo_raw"))
    ing.add_argument("--out", type=Path, default=Path("data/kurosiwo"))
    ing.add_argument("--events-file", type=Path, default=Path("configs/datasets/kurosiwo_events.yaml"))
    ing.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ing.add_argument("--overlay", type=Path, action="append", default=[])
    ing.add_argument("--event", action="append", default=None)
    ing.add_argument("--workers", type=int, default=4)
    rel = sub.add_parser("relabel-water", help="the same chips with label = flood ∪ hydro")
    rel.add_argument("--src", type=Path, default=Path("data/kurosiwo"))
    rel.add_argument("--dst", type=Path, default=Path("data/kurosiwo_water"))
    rel.add_argument("--raw", type=Path, default=Path("data/kurosiwo_raw"))
    rel.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    rel.add_argument("--overlay", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "download":
        download(args.raw)
        return 0
    from floodfusion.data.ingest import build_dataset

    cfg = load_config(args.config, overlays=args.overlay)
    if args.command == "ingest":
        report = ingest_kurosiwo(args.raw, args.out, cfg, args.events_file, events=args.event, workers=args.workers)
        LOGGER.info("summary: %s; excluded: %s", report["summary"], report["excluded"])
        LOGGER.info("manifest %s", build_dataset(args.out, cfg, split_epsg=SPLIT_EPSG))
        return 1 if report["summary"]["failed"] else 0
    n = relabel_water(args.src, args.dst, args.raw, cfg)
    LOGGER.info("%d stacks relabelled; manifest %s", n, build_dataset(args.dst, cfg, split_epsg=SPLIT_EPSG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
