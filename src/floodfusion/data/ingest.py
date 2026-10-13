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

Steps 3-6 and 9 are shared with the Sen1Floods11 route and live in
:mod:`floodfusion.data.assembly`; archive access is :mod:`floodfusion.data.archive`.

Memory scales at roughly 200 B per grid pixel: a 30 x 30 km AOI (9 Mpx) needs ~2 GB.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.archive import (
    GDAL_ENV,
    PC_STAC_URL,
    IngestSources,
    grd_id_for_rtc,
    items_by_id,
    items_in_bbox,
    product_annotation_href,
)
from floodfusion.data.assembly import (
    GridReference,
    GridTask,
    assemble_on_grid,
    incidence_on_grid,
    read_acquisition_geometry,
)
from floodfusion.data.chipping import AnalysisGrid, build_analysis_grid
from floodfusion.data.records import ChipRecord, load_manifest, write_manifest
from floodfusion.data.stac import (
    EventQuery,
    NoMatchingPairError,
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

DEFAULT_SPLIT_EPSG = 3035
#: ~200 B per grid pixel in memory: 5e7 px is ~10 GB — a mistyped bbox, not an AOI.
MAX_GRID_PIXELS = 50_000_000


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
    #: Pin the post-event acquisition (one named pass of a multi-pass event).
    relative_orbit: int | None = None
    orbit_state: str | None = None
    #: The pre-event reference must also predate this (a region flooded twice).
    reference_before: date | None = None
    #: Sentinel-2 is searched this many days either side of the flood window (0: the window only).
    optical_window_days: int = 0
    notes: str = ""

    def query(self) -> EventQuery:
        def at(d: date | None) -> datetime | None:
            return None if d is None else datetime.combine(d, datetime.min.time())

        return EventQuery(
            event_id=self.event_id, bbox=self.bbox,
            flood_start=at(self.flood_start), flood_end=at(self.flood_end),
            pre_event_window_days=self.pre_event_window_days,
            relative_orbit=self.relative_orbit, orbit_state=self.orbit_state,
            reference_before=at(self.reference_before),
            optical_start=at(self.flood_start - timedelta(days=self.optical_window_days)),
            optical_end=at(self.flood_end + timedelta(days=self.optical_window_days)),
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
    for key in ("flood_start", "flood_end", "reference_before"):
        if key not in values or values[key] is None:
            continue
        v = values[key]
        values[key] = v if isinstance(v, date) else date.fromisoformat(str(v))
    if values["flood_end"] < values["flood_start"]:
        raise ValueError(f"{path}: flood_end precedes flood_start")
    values["label_where"] = {k: tuple(v) if isinstance(v, list | tuple) else (v,)
                             for k, v in (values.get("label_where") or {}).items()}
    return EventConfig(**values)


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


def _reference_layers(sources: IngestSources, event: EventConfig, grid: AnalysisGrid) -> GridReference:
    """AOI, permanent water (JRC >= 80 % OR the CEMS reference water), land cover, label."""
    from floodfusion.data.sources import read_jrc_occurrence, read_worldcover
    from floodfusion.masks import permanent_water_from_occurrence

    if event.aoi_path is None:
        LOGGER.warning("%s: no aoi_path — every unlabelled pixel of the grid is scored as dry", event.event_id)
        inside_aoi = np.ones((grid.height, grid.width), dtype=bool)
    else:
        inside_aoi = rasterise_vector(event.aoi_path, grid, layer=event.aoi_layer)
    occurrence = read_jrc_occurrence(items_in_bbox(sources.client, "jrc-gsw", event.bbox), grid)
    permanent = permanent_water_from_occurrence(np.nan_to_num(occurrence, nan=0.0))
    if event.reference_water_path is not None:
        permanent = permanent | rasterise_vector(event.reference_water_path, grid, layer=event.reference_water_layer)
    worldcover = items_in_bbox(sources.client, "esa-worldcover", event.bbox)
    latest = max((str(i["properties"].get("start_datetime", "")) for i in worldcover), default="")
    landcover = read_worldcover([i for i in worldcover if str(i["properties"].get("start_datetime", "")) == latest], grid)
    # Flood = observed water AND NOT permanent water (docs/05 section 3.3).
    label = rasterise_vector(event.label_path, grid, layer=event.label_layer, where=event.label_where) & ~permanent
    return GridReference(label=label.astype(np.float64), inside_aoi=inside_aoi, permanent=permanent, landcover=landcover)


def ingest_event(
    event: EventConfig,
    out_root: Path | str,
    cfg: ExperimentConfig,
    *,
    sources: IngestSources | None = None,
    split_epsg: int = DEFAULT_SPLIT_EPSG,
) -> dict[str, Any]:
    """Ingest one event into ``<out_root>/events/<event_id>/``; returns its ``event.json``.

    GDAL's HTTP reads run with explicit timeouts and retries (:data:`floodfusion.data.archive.GDAL_ENV`).
    """
    import rasterio

    with rasterio.Env(**GDAL_ENV):
        return _ingest_event(event, out_root, cfg, sources=sources, split_epsg=split_epsg)


def _ingest_event(
    event: EventConfig,
    out_root: Path | str,
    cfg: ExperimentConfig,
    *,
    sources: IngestSources | None,
    split_epsg: int,
) -> dict[str, Any]:
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
    if not matched.dem_ids:
        raise ValueError(f"{event.event_id}: no {profile.dem_collection} tiles found for the AOI "
                         "(see the DEM search warning above)")
    task = GridTask(
        event_id=event.event_id, out_dir=event_dir, grid=grid, matched=matched,
        annotations=tuple(read_acquisition_geometry(sources, matched)),
        reference=_reference_layers(sources, event, grid), basins=_event_basins(event, grid),
        dem_items=tuple(items_by_id(sources.client, profile.dem_collection, matched.dem_ids)),
        extra_meta={"event_config": {k: str(v) for k, v in vars(event).items()}},
        max_s2_candidates=event.max_s2_candidates,
    )
    return assemble_on_grid(task, cfg, sources, split_epsg=split_epsg)


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
        # event.json is written LAST: its presence means the whole ingest, QA included, finished.
        if (args.out / "events" / event.event_id / "event.json").is_file() and not args.force:
            LOGGER.info("%s already ingested (use --force to redo)", event.event_id)
            continue
        try:
            meta = ingest_event(event, args.out, cfg, split_epsg=args.split_epsg)
            LOGGER.info("%s: %d chips, flood %.1f %% of valid pixels", event.event_id, meta["n_chips"],
                        100 * meta["flood_fraction_of_valid"])
        except NoMatchingPairError as exc:
            # docs/05 2.1 criterion 3: an event without a same-orbit pre scene is DROPPED.
            LOGGER.error("%s dropped: %s", event.event_id, exc)
            failures += 1
        except Exception as exc:  # noqa: BLE001 - one event's failure must not abort the batch
            # Error texts of HTTP/GDAL failures embed signed URLs: never print the token.
            LOGGER.error("%s failed: %s: %s", event.event_id, type(exc).__name__, redact(str(exc)))
            failures += 1
    if not any((args.out / "events").glob("*/chips.parquet")):
        LOGGER.error("no event produced chips: nothing to merge into a manifest")
        return 1
    digest = build_dataset(args.out, cfg, split_epsg=args.split_epsg)
    LOGGER.info("manifest %s", digest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
