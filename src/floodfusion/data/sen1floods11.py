r"""Sen1Floods11 (Bonafilia et al., CVPRW 2020), re-ingested through floodfusion's own pipeline.

Sen1Floods11 offers 446 hand-labelled 512 x 512 chips from 11 flood events with a
Sentinel-1 GRD and a Sentinel-2 **L1C** image each. This model needs more than that: a
**pre-event** Sentinel-1 reference on the same relative orbit (the change term), Sentinel-2
**L2A** with its scene classification (the quality gate), and the terrain layers. So only
the **labels** are taken from Sen1Floods11; every input is fetched again through the same
code that ingests a CEMS activation (:mod:`floodfusion.data.assembly`):

- the event's named acquisition — relative orbit and date from the dataset metadata — is
  matched to Planetary Computer ``sentinel-1-rtc`` together with the latest same-orbit
  pre-event acquisition within 60 days;
- Sentinel-2 L2A on the dataset's optical date, GLO-30, JRC GSW, WorldCover;
- each chip becomes a small stack on its own 10 m UTM grid; terrain is computed with a
  margin so shadow, layover and HAND see beyond the 5 km chip.

**Split groups.** Every chip of an event belongs to one group (``S1F11_<event>``), the
union of the event's chip footprints. Events lie on different continents, so grouping by
event is the leakage-free blocking unit: leave-one-event-out.

**The official split is not blocked.** Its train, validation and test chips are drawn
from the *same* ten events; only Bolivia is held out (:func:`official_split`). Scores on
it are therefore within-event scores — the spatial-leakage row of the ablation, on a
public leaderboard.

**Licence.** The dataset declares no licence. Nothing derived from it is redistributed:
:func:`download` fetches the labels from the official bucket into a local directory.

**Geometry.** Measured on this archive: the Sentinel-2 L2A passes of 2016-2019 sit about
2 px (~20 m) from the RTC gamma0 and DEM, and the hand labels follow the optical image to
0.0 px. Left alone, that offset handicaps every SAR-only model against a label drawn on
the optical image. The optical layers **and the label** are therefore moved, by one
whole-pixel shift per event measured from the pooled windows of many chips, onto the
SAR/DEM geometry (:func:`floodfusion.data.assembly.align_optical_to_sar`). A mixed
Sentinel-1A/1B pre/post pair shows a further ~0.6 px azimuth offset; same-satellite pairs
co-register to 0.0 px.

**Labels.** ``1`` water, ``0`` not water, ``-1`` unlabelled (``NaN`` here). Water includes
permanent water; it is removed with JRC occurrence >= 80 %, so the label becomes *flood*.
The labels were drawn mostly on the Sentinel-2 image, so water under canopy is
under-represented.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.archive import (
    GDAL_ENV,
    PC_STAC_URL,
    IngestSources,
    items_by_id,
    items_in_bbox,
)
from floodfusion.data.assembly import (
    OPTICAL_SHIFT_MIN_PX,
    SHIFT_MAX_CI_PX,
    GridReference,
    GridTask,
    assemble_on_grid,
    expand_grid,
    label_optical_window_shifts,
    label_window_shifts,
    pooled_shift,
    read_acquisition_geometry,
    sar_optical_window_shifts,
)
from floodfusion.data.chipping import AnalysisGrid, build_analysis_grid
from floodfusion.data.http import redacted_traceback
from floodfusion.data.records import atomic_write_bytes
from floodfusion.data.stac import (
    PROFILES,
    EventQuery,
    InMemoryCatalogue,
    MatchedScenes,
    NoMatchingPairError,
    redact,
    search_matched_pairs,
)

__all__ = [
    "BUCKET_URL",
    "HandChip",
    "S1F11Event",
    "chip_reference",
    "chip_shift_windows",
    "choose_event_shift",
    "download",
    "estimate_event_shift",
    "ingest_sen1floods11",
    "list_hand_chips",
    "main",
    "official_split",
    "parse_metadata",
    "parse_split_csv",
]

LOGGER = logging.getLogger(__name__)

_HOST = "storage.googleapis.com"
_BUCKET = "sen1floods11"
_VERSION = "v1.1"
BUCKET_URL = f"https://{_HOST}/{_BUCKET}/{_VERSION}"
_LIST_URL = f"https://{_HOST}/storage/v1/b/{_BUCKET}/o"
_ALLOWED_PATH_PREFIXES = (f"/{_BUCKET}/", f"/storage/v1/b/{_BUCKET}/")
_MAX_OBJECT_BYTES = 16 * 1024 * 1024
_HTTP_TIMEOUT_S = 60
_METADATA = "Sen1Floods11_Metadata.geojson"
_HAND = "data/flood_events/HandLabeled"
SPLIT_FILES = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "val": "splits/flood_handlabeled/flood_valid_data.csv",
    "test": "splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": "splits/flood_handlabeled/flood_bolivia_data.csv",
}
#: Chip file prefixes that differ from the metadata ``location``.
_PREFIX_BY_LOCATION = {"Cambodia": "Mekong"}
GROUP_PREFIX = "S1F11_"
#: One metric CRS for every chip centroid: the events span five continents, and the split only
#: needs distances between chips of the same event (events are thousands of km apart).
SPLIT_EPSG = 3857
#: An int16 value the labels (-1, 0, 1) never take; marks off-chip pixels during the warp.
_LABEL_SENTINEL = -32768
#: 256 px = 2.56 km of terrain context around each 5 km chip (shadow, layover, HAND).
DEFAULT_MARGIN_PX = 256
#: The dataset's date is a calendar date; the same orbit cannot repeat within a day.
_DATE_TOLERANCE_DAYS = 1
#: Chips read per event to measure its misregistration: every chip (the largest event has 69).
_SHIFT_SAMPLE_CHIPS = 96
#: An event-level offset is a measurement only from this many chips: its interval resamples
#: chips, and with few of them the bootstrap under-covers (~70 % nominal-95 % coverage at 5).
_MIN_SHIFT_GRIDS = 12
#: Reflectance band index of SWIR (B11) in the (B3, B8, B11) stack.
_SWIR = 2


# ---------------------------------------------------------------------------
# Download (labels, metadata, splits) — allow-listed, size-capped, md5-verified
# ---------------------------------------------------------------------------
def _check_url(url: str) -> str:
    from floodfusion.data.http import check_https_url

    return check_https_url(url, hosts=(_HOST,), path_prefixes=_ALLOWED_PATH_PREFIXES)


def _http_get(url: str, *, params: Mapping[str, str] | None = None) -> bytes:
    from floodfusion.data.http import capped_get

    return capped_get(url, check=_check_url, cap=_MAX_OBJECT_BYTES, params=params, timeout_s=_HTTP_TIMEOUT_S,
                      retries=3)


Getter = Callable[..., bytes]


def _list_objects(prefix: str, get: Getter) -> list[dict[str, str]]:
    """Every object under ``prefix`` (``name``, ``size``, ``md5Hash``), following pagination."""
    out: list[dict[str, str]] = []
    token: str | None = None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        page = json.loads(get(_LIST_URL, params=params))
        out.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return out


def _md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode("ascii")


def download(raw_root: Path | str, *, layers: Sequence[str] = ("LabelHand",), get: Getter = _http_get) -> dict[str, int]:
    """Fetch metadata, the hand-labelled split lists and ``layers`` into ``raw_root``.

    Resumable: a file already present with the listed md5 is skipped. Every download is
    verified against the bucket's md5 before it is written (atomically).
    """
    root = Path(raw_root)
    splits = {f"{_VERSION}/{rel}" for rel in SPLIT_FILES.values()}
    wanted = [o for o in _list_objects(f"{_VERSION}/{_METADATA}", get) if o["name"] == f"{_VERSION}/{_METADATA}"]
    wanted += [o for o in _list_objects(f"{_VERSION}/splits/flood_handlabeled/", get) if o["name"] in splits]
    if len(wanted) != 1 + len(splits):
        raise OSError(f"bucket listing is missing the metadata or split files: got {[o['name'] for o in wanted]}")
    for layer in layers:
        if "/" in layer or layer.startswith("."):
            raise ValueError(f"invalid layer name {layer!r}")
        wanted += _list_objects(f"{_VERSION}/{_HAND}/{layer}/", get)
    counts = {"fetched": 0, "skipped": 0}
    for obj in wanted:
        rel = obj["name"].removeprefix(f"{_VERSION}/")
        dest = (root / rel).resolve()
        if root.resolve() not in dest.parents or ".." in Path(rel).parts:
            raise ValueError(f"object name escapes the download root: {obj['name']!r}")
        if not obj.get("md5Hash"):
            raise OSError(f"bucket listing has no md5 for {obj['name']}; refusing an unverifiable download")
        if dest.is_file() and _md5_b64(dest.read_bytes()) == obj["md5Hash"]:
            counts["skipped"] += 1
            continue
        data = get(f"https://{_HOST}/{_BUCKET}/{obj['name']}")
        if _md5_b64(data) != obj["md5Hash"]:
            raise OSError(f"md5 mismatch for {obj['name']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(dest, data)
        counts["fetched"] += 1
    LOGGER.info("Sen1Floods11: %d fetched, %d already present", counts["fetched"], counts["skipped"])
    return counts


# ---------------------------------------------------------------------------
# Metadata, chips, splits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class S1F11Event:
    """One Sen1Floods11 event: the acquisition its labels were drawn on."""

    name: str              # chip-file prefix ("Mekong", "Sri-Lanka", ...)
    location: str          # metadata location ("Cambodia", ...)
    s1_date: date
    s2_date: date
    relative_orbit: int
    orbit_state: str

    @property
    def group(self) -> str:
        return f"{GROUP_PREFIX}{self.name}"


def parse_metadata(path: Path | str) -> dict[str, S1F11Event]:
    """``{chip-file prefix: event}`` from ``Sen1Floods11_Metadata.geojson`` (duplicates collapse)."""
    def day(text: str) -> date:
        return date(*map(int, str(text).split("/")))

    events: dict[str, S1F11Event] = {}
    for feature in json.loads(Path(path).read_text(encoding="utf-8"))["features"]:
        p = feature["properties"]
        name = _PREFIX_BY_LOCATION.get(p["location"], p["location"])
        event = S1F11Event(name, p["location"], day(p["s1_date"]), day(p["s2_date"]),
                           int(p["rel_orbit_num"]), str(p["orbit"]).lower())
        if name in events and events[name] != event:
            raise ValueError(f"conflicting metadata rows for {name}: {events[name]} vs {event}")
        events[name] = event
    return events


def parse_split_csv(text: str) -> tuple[str, ...]:
    """Chip ids (``Ghana_103272``) from an official split file (``<id>_S1Hand.tif,<id>_LabelHand.tif``)."""
    ids = []
    for line in text.splitlines():
        first = line.split(",", 1)[0].strip()
        if first:
            ids.append(first.rsplit("_", 1)[0])
    return tuple(ids)


def official_split(raw_root: Path | str) -> dict[str, str]:
    """``{chip id: "train" | "val" | "test" | "bolivia"}`` of the dataset's own hand-labelled split."""
    out: dict[str, str] = {}
    for split, rel in SPLIT_FILES.items():
        for chip in parse_split_csv((Path(raw_root) / rel).read_text(encoding="utf-8")):
            if out.setdefault(chip, split) != split:
                raise ValueError(f"{chip} is in both {out[chip]} and {split}")
    return out


@dataclass(frozen=True)
class HandChip:
    chip_id: str          # "Ghana_103272"
    event: str            # "Ghana"
    label_path: Path

    @property
    def stack_id(self) -> str:
        return f"{GROUP_PREFIX}{self.chip_id}"


def list_hand_chips(raw_root: Path | str) -> list[HandChip]:
    chips = []
    for path in sorted((Path(raw_root) / _HAND / "LabelHand").glob("*_LabelHand.tif")):
        chip_id = path.name.removesuffix("_LabelHand.tif")
        chips.append(HandChip(chip_id, chip_id.rsplit("_", 1)[0], path))
    return chips


def _footprint(path: Path) -> tuple[float, float, float, float]:
    import rasterio
    from rasterio.warp import transform_bounds

    # GTiff only: a downloaded "*.tif" holding VRT XML would otherwise make GDAL read other files.
    with rasterio.open(path, driver="GTiff") as src:
        if src.crs is None:
            raise ValueError(f"{path.name}: label has no CRS")
        w, s_, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return (w, s_, e, n)


# ---------------------------------------------------------------------------
# Reference layers of one chip
# ---------------------------------------------------------------------------
def chip_reference(label_path: Path, grid: AnalysisGrid, jrc_items: Sequence[Mapping[str, Any]],
                   worldcover_items: Sequence[Mapping[str, Any]]) -> GridReference:
    """The hand label on ``grid``: water minus permanent water; unlabelled and off-chip are NaN."""
    from floodfusion.data.chipping import reproject_to_grid
    from floodfusion.data.sources import read_jrc_occurrence, read_worldcover
    from floodfusion.masks import permanent_water_from_occurrence

    # The label GeoTIFFs declare no nodata: without a sentinel the warp fills off-chip pixels
    # with 0 — "not water" — and scores the grid's corners as dry land.
    raw = reproject_to_grid(str(label_path), grid, kind="categorical", src_nodata=_LABEL_SENTINEL, dtype="float32")
    inside = np.isfinite(raw)
    if jrc_items:
        occurrence = read_jrc_occurrence(jrc_items, grid)
        permanent = permanent_water_from_occurrence(np.nan_to_num(occurrence, nan=0.0))
    else:
        LOGGER.warning("%s: no JRC GSW tile — no permanent water removed from the label", label_path.name)
        permanent = np.zeros((grid.height, grid.width), dtype=bool)
    label = np.where(raw == 1, 1.0, np.where(raw == 0, 0.0, np.nan))
    label = np.where(permanent & (label == 1), 0.0, label)
    landcover = (read_worldcover(worldcover_items, grid) if worldcover_items
                 else np.zeros((grid.height, grid.width), dtype=np.uint8))
    return GridReference(label=label, inside_aoi=inside, permanent=permanent, landcover=landcover)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _union_bbox(boxes: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    arr = np.asarray(list(boxes), dtype=np.float64)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())


def _grow_bbox(bbox: Sequence[float], metres: float) -> tuple[float, float, float, float]:
    lat = np.radians(0.5 * (bbox[1] + bbox[3]))
    dlat = metres / 111_320.0
    dlon = metres / (111_320.0 * max(np.cos(lat), 0.05))
    return bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat


def _event_query(event: S1F11Event, bbox: Sequence[float]) -> EventQuery:
    tol = timedelta(days=_DATE_TOLERANCE_DAYS)
    when = datetime.combine(event.s1_date, datetime.min.time())
    optical = datetime.combine(event.s2_date, datetime.min.time())
    return EventQuery(event.group, tuple(bbox), when - tol, when + tol, pre_event_window_days=60,
                      relative_orbit=event.relative_orbit, orbit_state=event.orbit_state,
                      optical_start=optical, optical_end=optical)


def _restrict(matched: MatchedScenes, by_id: Mapping[str, Mapping[str, Any]], bbox: Sequence[float]) -> MatchedScenes:
    """The match narrowed to the frames and tiles that touch one chip."""
    def keep(ids: Sequence[str]) -> list[int]:
        return [k for k, i in enumerate(ids) if _overlaps(by_id[i]["bbox"], bbox)]

    post = [matched.s1_post_frames[k] for k in keep(matched.s1_post_frames)]
    pre = [matched.s1_pre_frames[k] for k in keep(matched.s1_pre_frames)]
    if not post or not pre:
        raise NoMatchingPairError(f"no {'post' if not post else 'pre'}-event frame covers the chip")
    s2 = keep(matched.s2_ids)
    return replace(
        matched, s1_post_frames=tuple(post), s1_pre_frames=tuple(pre),
        s2_ids=tuple(matched.s2_ids[k] for k in s2),
        s2_cloud_cover=tuple(matched.s2_cloud_cover[k] for k in s2),
        s2_datetimes=tuple(matched.s2_datetimes[k] for k in s2) if matched.s2_datetimes else (),
    )


@dataclass(frozen=True)
class _EventArchive:
    """Every item one event's chips need, fetched once and searched in memory."""

    matched: MatchedScenes
    annotations: tuple[Any, ...]
    catalogue: InMemoryCatalogue
    by_id: dict[str, dict[str, Any]]
    dem: tuple[dict[str, Any], ...]
    jrc: tuple[dict[str, Any], ...]
    worldcover: tuple[dict[str, Any], ...]


def _event_archive(event: S1F11Event, bbox: Sequence[float], sources: IngestSources, margin_m: float) -> _EventArchive:
    profile = PROFILES["planetary_computer"]
    matched = search_matched_pairs(_event_query(event, bbox), PC_STAC_URL, client=sources.client, profile=profile)
    annotations = tuple(read_acquisition_geometry(sources, matched))
    wide = _grow_bbox(bbox, margin_m)
    s1 = items_by_id(sources.client, profile.s1_collection, matched.s1_post_frames + matched.s1_pre_frames)
    s2 = items_by_id(sources.client, profile.s2_collection, matched.s2_ids)
    dem = items_in_bbox(sources.client, profile.dem_collection, wide)
    jrc = items_in_bbox(sources.client, "jrc-gsw", wide)
    worldcover = items_in_bbox(sources.client, "esa-worldcover", wide)
    latest = max((str(i["properties"].get("start_datetime", "")) for i in worldcover), default="")
    worldcover = [i for i in worldcover if str(i["properties"].get("start_datetime", "")) == latest]
    everything = s1 + s2 + dem + jrc + worldcover
    return _EventArchive(matched, annotations, InMemoryCatalogue.from_dicts(everything),
                         {i["id"]: i for i in everything}, tuple(dem), tuple(jrc), tuple(worldcover))


def _event_polygon(footprints: Iterable[Sequence[float]]) -> Any:
    from shapely.geometry import box
    from shapely.ops import unary_union

    return unary_union([box(*b) for b in footprints])


def _to_grid_crs(geometry: Any, epsg: int) -> Any:
    import shapely
    from pyproj import Transformer

    to_grid = Transformer.from_crs(4326, epsg, always_xy=True)
    return shapely.transform(geometry, lambda x, y: to_grid.transform(x, y), interleaved=False)


def choose_event_shift(estimates: Mapping[str, Mapping[str, Any]], *, max_ci_px: float = SHIFT_MAX_CI_PX,
                       min_grids: int = _MIN_SHIFT_GRIDS) -> dict[str, Any]:
    """The whole-pixel shifts that move an event's labels and Sentinel-2 onto the SAR geometry.

    ``estimates`` holds three pooled offsets (:func:`floodfusion.data.assembly.pooled_shift`):
    ``optical`` (SWIR vs SAR, clear windows), ``label`` (label water edges vs SAR) and
    ``label_vs_optical`` (label vs SWIR — no SAR involved). An estimate is used only if it is a
    measurement: at least ``min_grids`` chips and a 95 % interval within ``max_ci_px``.

    - Labels drawn on the optical image (``label_vs_optical`` not a measured offset of 1 px or
      more — ~0 px on 8 of 10 events) move with it by one shift: the label estimate if it is a
      measurement, else the optical one, else none. Clouds leave the SWIR estimate too few clear
      windows on monsoon events (India, Mekong, Bolivia, Sri Lanka kept labels 1-3 px off the SAR
      with it alone); the label's water edges measure the same offset.
    - Labels measured to sit elsewhere get their own shift; the optical keeps its own.

    The label estimate uses labels — of the held-out event too — for this two-parameter,
    model-independent registration; DATA_CARD.md discloses what that can and cannot bias.
    """
    def measured(name: str) -> bool:
        e = estimates.get(name) or {}
        return e.get("n_grids", 0) >= min_grids and float(e.get("ci_halfwidth_px", float("nan"))) <= max_ci_px

    def shift(name: str) -> list[int]:
        return [int(v) for v in estimates[name]["shift_px"]]

    follows = not (measured("label_vs_optical")
                   and math.hypot(*estimates["label_vs_optical"]["median_shift"]) >= OPTICAL_SHIFT_MIN_PX)
    if follows:
        source = "label" if measured("label") else "optical" if measured("optical") else "none"
        common = [0, 0] if source == "none" else shift(source)
        label_shift, label_source, optical_shift, optical_source = common, source, common, source
    else:
        label_source = "label" if measured("label") else "none"
        optical_source = "optical" if measured("optical") else "none"
        label_shift = shift("label") if label_source == "label" else [0, 0]
        optical_shift = shift("optical") if optical_source == "optical" else [0, 0]
    return {"label_shift_px": label_shift, "label_source": label_source,
            "optical_shift_px": optical_shift, "optical_source": optical_source,
            "labels_follow_optical": follows, "estimates": {k: dict(v) for k, v in estimates.items()}}


def chip_shift_windows(vv_post_db: np.ndarray, label_raw: np.ndarray,
                       reflectance_clear: np.ndarray | None) -> dict[str, np.ndarray]:
    """One chip's window shifts for the three event-level offsets (cloudy optical: NaN)."""
    label = np.where(label_raw == 1, 1.0, np.where(label_raw == 0, 0.0, np.nan))
    out = {"label": label_window_shifts(vv_post_db, label)}
    if reflectance_clear is not None and np.isfinite(reflectance_clear[_SWIR]).mean() > 0.2:
        out["optical"] = sar_optical_window_shifts(vv_post_db, reflectance_clear)
        out["label_vs_optical"] = label_optical_window_shifts(reflectance_clear, label)
    return out


def estimate_event_shift(per_chip: Sequence[Mapping[str, np.ndarray]], *, n_read_failed: int = 0) -> dict[str, Any]:
    """Pool the chips' window shifts and decide (:func:`choose_event_shift`); JSON-ready record."""
    estimates = {k: pooled_shift([c[k] for c in per_chip if k in c])
                 for k in ("optical", "label", "label_vs_optical")}
    record = {**choose_event_shift(estimates), "n_chips_read": len(per_chip), "n_read_failed": int(n_read_failed)}
    return _json_ready(record)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _chip_shift_inputs(chip: HandChip, footprint: Sequence[float], archive: _EventArchive,
                       cfg: ExperimentConfig) -> dict[str, np.ndarray] | None:
    import rasterio

    from floodfusion.data.chipping import CLEAR_QUALITY, reproject_to_grid
    from floodfusion.data.sources import read_s1_rtc, read_s2_reflectance
    from floodfusion.masks import quality_from_scl

    try:
        matched = _restrict(archive.matched, archive.by_id, footprint)
    except NoMatchingPairError:
        return None
    grid = build_analysis_grid(tuple(footprint), cfg.grid.pixel_size_m)
    with rasterio.Env(**GDAL_ENV):  # per worker thread: bounded HTTP timeouts
        vv, _ = read_s1_rtc(items_by_id(archive.catalogue, PROFILES["planetary_computer"].s1_collection,
                                        matched.s1_post_frames), grid, "VV")
        raw = reproject_to_grid(str(chip.label_path), grid, kind="categorical", src_nodata=_LABEL_SENTINEL,
                                dtype="float32")
        rho = None
        if matched.s2_ids:
            rho, scl, _ = read_s2_reflectance(items_by_id(archive.catalogue, "sentinel-2-l2a", matched.s2_ids), grid)
            rho = np.where(quality_from_scl(scl)[None] >= CLEAR_QUALITY, rho, np.nan)
    return chip_shift_windows(vv, raw, rho)


def _event_shift(chips: Sequence[HandChip], footprints: Mapping[str, Sequence[float]],
                 archive: _EventArchive, cfg: ExperimentConfig, workers: int) -> dict[str, Any]:
    """Measure an event's offsets on up to :data:`_SHIFT_SAMPLE_CHIPS` chips and decide its shifts.

    Geolocation error of an optical pass is a property of the pass (attitude, ephemeris), not
    of a 5 km chip: offsets are pooled over chips and one decision applies to every chip. A chip
    whose reads fail is counted and skipped; the event fails only if every read fails.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Evenly spread over the event (ids sort roughly by place): not the first N chips.
    picks = np.unique(np.linspace(0, len(chips) - 1, min(len(chips), _SHIFT_SAMPLE_CHIPS)).round().astype(int))

    def read(chip: HandChip) -> tuple[dict[str, np.ndarray] | None, bool]:
        try:
            return _chip_shift_inputs(chip, footprints[chip.chip_id], archive, cfg), False
        except Exception as exc:  # noqa: BLE001 - one unreadable chip must not decide the event's shift
            LOGGER.warning("%s: shift inputs unreadable, skipped: %s", chip.chip_id, redact(str(exc))[:200])
            return None, True

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        results = list(pool.map(read, [chips[int(i)] for i in picks]))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    per_chip = [r for r, _ in results if r is not None]
    failed = sum(f for _, f in results)
    if failed and not per_chip:
        raise RuntimeError(f"every shift read failed ({failed} chips)")
    record = estimate_event_shift(per_chip, n_read_failed=failed)
    for name, est in record["estimates"].items():
        LOGGER.info("%s: %s offset %s px (95%% +/- %s) from %d windows on %d chips", chips[0].event, name,
                    est["median_shift"], est["ci_halfwidth_px"], est["n_windows"], est["n_grids"])
    LOGGER.info("%s: labels move by %s (%s), optical by %s (%s)%s", chips[0].event, record["label_shift_px"],
                record["label_source"], record["optical_shift_px"], record["optical_source"],
                "" if record["labels_follow_optical"] else "; the labels do not follow the optical image")
    return record


def _stored_event_shift(out_root: Path, chips: Sequence[HandChip]) -> dict[str, Any] | None:
    """The decision recorded by an earlier run of this event: a resumed event keeps its shift."""
    for chip in chips:
        path = out_root / "events" / chip.stack_id / "event.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8")).get("event_shift")
            if isinstance(record, dict) and "label_shift_px" in record:
                return record
    return None


def _ingest_chip(chip: HandChip, event: S1F11Event, archive: _EventArchive, footprint: Sequence[float],
                 event_polygon: Any, out_root: Path, cfg: ExperimentConfig, sources: IngestSources,
                 margin_px: int, split_epsg: int, event_shift: Mapping[str, Any]) -> dict[str, Any]:
    import rasterio

    # Every GDAL read of this chip — reference layers included — runs with bounded HTTP
    # timeouts: rasterio.Env is per thread, and this runs in a worker thread.
    with rasterio.Env(**GDAL_ENV):
        return _assemble_chip(chip, event, archive, footprint, event_polygon, out_root, cfg, sources, margin_px,
                              split_epsg, event_shift)


def _assemble_chip(chip: HandChip, event: S1F11Event, archive: _EventArchive, footprint: Sequence[float],
                   event_polygon: Any, out_root: Path, cfg: ExperimentConfig, sources: IngestSources,
                   margin_px: int, split_epsg: int, event_shift: Mapping[str, Any]) -> dict[str, Any]:
    from rasterio.warp import transform_bounds

    grid = build_analysis_grid(tuple(footprint), cfg.grid.pixel_size_m)
    wide = transform_bounds(f"EPSG:{grid.epsg}", "EPSG:4326", *expand_grid(grid, margin_px).bounds)
    matched = _restrict(archive.matched, archive.by_id, footprint)
    task = GridTask(
        event_id=chip.stack_id, out_dir=out_root / "events" / chip.stack_id, grid=grid, matched=matched,
        annotations=archive.annotations,
        reference=chip_reference(chip.label_path, grid, [i for i in archive.jrc if _overlaps(i["bbox"], footprint)],
                                 [i for i in archive.worldcover if _overlaps(i["bbox"], footprint)]),
        basins={event.group: _to_grid_crs(event_polygon, grid.epsg)},
        dem_items=tuple(i for i in archive.dem if _overlaps(i["bbox"], wide)),
        extra_meta={"sen1floods11": {"chip_id": chip.chip_id, **{k: str(v) for k, v in asdict(event).items()}},
                    "event_shift": dict(event_shift)},
        terrain_margin_px=margin_px, label_follows_optical=True,
        optical_shift_px=(int(event_shift["optical_shift_px"][0]), int(event_shift["optical_shift_px"][1])),
        label_shift_px=(None if event_shift["labels_follow_optical"]
                        else (int(event_shift["label_shift_px"][0]), int(event_shift["label_shift_px"][1]))),
        generator="floodfusion.data.sen1floods11 (labels: Sen1Floods11 v1.1; inputs: Planetary Computer)",
    )
    chip_sources = IngestSources(client=archive.catalogue, fetch_bytes=sources.fetch_bytes)
    return assemble_on_grid(task, cfg, chip_sources, split_epsg=split_epsg)


def _ingest_one_event(event: S1F11Event, chips: Sequence[HandChip], out_root: Path, cfg: ExperimentConfig,
                      sources: IngestSources, margin_px: int, split_epsg: int, workers: int) -> dict[str, str]:
    from concurrent.futures import ThreadPoolExecutor

    todo = [c for c in chips if not (out_root / "events" / c.stack_id / "event.json").is_file()]
    status: dict[str, Any] = {c.chip_id: "done-before" for c in chips if c not in todo}
    stored = _stored_event_shift(out_root, chips)
    if stored is not None:
        status["event_shift"] = stored
    if not todo:
        return status
    margin_m = margin_px * cfg.grid.pixel_size_m
    footprints: dict[str, tuple[float, float, float, float]] = {}
    for chip in chips:  # an unreadable label fails its chip, not the event
        try:
            footprints[chip.chip_id] = _footprint(chip.label_path)
        except Exception as exc:  # noqa: BLE001
            status[chip.chip_id] = f"failed: unreadable label: {type(exc).__name__}: {str(exc)[:200]}"
            LOGGER.error("%s: unreadable label %s", chip.chip_id, chip.label_path)
    chips = [c for c in chips if c.chip_id in footprints]
    todo = [c for c in todo if c.chip_id in footprints]
    if not todo:
        return status
    try:
        archive = _event_archive(event, _union_bbox(footprints.values()), sources, margin_m)
        polygon = _event_polygon(footprints.values())
        if stored is not None:
            LOGGER.info("%s: resuming with the shift recorded by the first run", event.group)
        shift = stored if stored is not None else _event_shift(chips, footprints, archive, cfg, workers)
    except NoMatchingPairError as exc:
        LOGGER.error("%s dropped: %s", event.group, exc)
        return {**status, **{c.chip_id: f"dropped: {exc}" for c in todo}}
    except Exception as exc:  # noqa: BLE001 - one event's failure must not abort the dataset
        message = f"failed: {type(exc).__name__}: {redact(str(exc))[:300]}"
        LOGGER.error("%s %s\n%s", event.group, message, redacted_traceback(exc))
        return {**status, **{c.chip_id: message for c in todo}}
    status["event_shift"] = shift

    def run(chip: HandChip) -> tuple[str, str]:
        try:
            meta = _ingest_chip(chip, event, archive, footprints[chip.chip_id], polygon, out_root, cfg, sources,
                                margin_px, split_epsg, shift)
            if not meta["n_chips"]:
                return chip.chip_id, f"dropped: no window passed the filters {meta['chip_counts']}"
            verdict = meta["coregistration_qa"]["p1_m1_pass"]
            qa = "pass" if verdict is True else "fail" if verdict is False else "not measurable"
            return chip.chip_id, f"ok: {meta['n_chips']} windows, optical={meta['has_optical']}, qa={qa}"
        except NoMatchingPairError as exc:
            return chip.chip_id, f"dropped: {exc}"
        except Exception as exc:  # noqa: BLE001 - one chip's failure must not abort the event
            LOGGER.error("%s failed\n%s", chip.chip_id, redacted_traceback(exc))
            return chip.chip_id, f"failed: {type(exc).__name__}: {redact(str(exc))}"

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for chip_id, outcome in pool.map(run, todo):
            status[chip_id] = outcome
            LOGGER.info("%s %s", chip_id, outcome)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C: do not finish the queued chips first
        raise
    pool.shutdown()
    return status


def ingest_sen1floods11(
    raw_root: Path | str,
    out_root: Path | str,
    cfg: ExperimentConfig,
    *,
    events: Sequence[str] | None = None,
    sources: IngestSources | None = None,
    margin_px: int = DEFAULT_MARGIN_PX,
    split_epsg: int = SPLIT_EPSG,
    workers: int = 4,
) -> dict[str, Any]:
    """Re-ingest every hand-labelled chip (or those of ``events``); write ``ingest_report.json``.

    Resumable per chip. ``split_epsg`` defaults to Web Mercator: the events span five
    continents, and the split only needs one metric space for the buffer test between
    chips of the *same* event (events themselves are thousands of km apart).
    """
    raw_root, out_root = Path(raw_root), Path(out_root)
    metadata = parse_metadata(raw_root / _METADATA)
    chips = list_hand_chips(raw_root)
    names = sorted({c.event for c in chips} if events is None else set(events))
    unknown = sorted(set(names) - set(metadata))
    if unknown:
        raise ValueError(f"no metadata for events {unknown}; known: {sorted(metadata)}")
    sources = sources or IngestSources.open(PC_STAC_URL)
    report: dict[str, Any] = {"events": {}}
    for name in names:
        event_chips = [c for c in chips if c.event == name]
        LOGGER.info("%s: %d chips, orbit %d %s, S1 %s, S2 %s", name, len(event_chips),
                    metadata[name].relative_orbit, metadata[name].orbit_state, metadata[name].s1_date,
                    metadata[name].s2_date)
        report["events"][name] = _ingest_one_event(metadata[name], event_chips, out_root, cfg, sources,
                                                   margin_px, split_epsg, workers)
    outcomes = [o for ev in report["events"].values() for k, o in ev.items() if k != "event_shift"]
    report["summary"] = {k: sum(o.startswith(k) for o in outcomes) for k in ("ok", "done-before", "dropped", "failed")}
    # P1-M1 co-registration verdicts of the grids ingested in this run ("not measurable" is not "failed").
    report["qa"] = {k: sum(o.endswith(f"qa={k}") for o in outcomes) for k in ("pass", "fail", "not measurable")}
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_root / "ingest_report.json", json.dumps(report, indent=2).encode("utf-8"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sen1Floods11 labels, re-ingested through floodfusion.")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch labels, metadata and split lists (md5-verified, resumable)")
    dl.add_argument("--raw", type=Path, default=Path("data/sen1floods11_raw"))
    dl.add_argument("--layer", action="append", default=None, help="hand-labelled layer (default LabelHand)")
    ing = sub.add_parser("ingest", help="re-fetch every input from Planetary Computer and build the chip set")
    ing.add_argument("--raw", type=Path, default=Path("data/sen1floods11_raw"))
    ing.add_argument("--out", type=Path, default=Path("data/sen1floods11"))
    ing.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ing.add_argument("--overlay", type=Path, action="append", default=[])
    ing.add_argument("--event", action="append", default=None, help="only this event (repeatable)")
    ing.add_argument("--margin-px", type=int, default=DEFAULT_MARGIN_PX)
    ing.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "download":
        download(args.raw, layers=tuple(args.layer or ("LabelHand",)))
        return 0
    from floodfusion.data.ingest import build_dataset

    cfg = load_config(args.config, overlays=args.overlay)
    report = ingest_sen1floods11(args.raw, args.out, cfg, events=args.event, margin_px=args.margin_px,
                                 workers=args.workers)
    LOGGER.info("summary: %s", report["summary"])
    digest = build_dataset(args.out, cfg, split_epsg=SPLIT_EPSG)
    LOGGER.info("manifest %s", digest)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
