r"""GEOID-Flood (Chiriaco et al., ECCV-W 2026): CEMS flood labels of 219 activations, re-ingested to train v1.2.

GEOID-Flood labels Copernicus EMS rapid-mapping delineations in three manually validated classes on 10 m UTM tiles of
1024 × 1024 px:

- ``0``: background;
- ``1``: permanent water;
- ``2``: flooded water;
- ``255``: not mapped.

Like Kuro Siwo, only the labels are used. Every input is fetched again from Planetary Computer through the shared
re-ingest (:mod:`floodfusion.data.kurosiwo`), so the two sets get the same passes, registration rule, assembly and
chips.

**Which events.** Only activations before EMSR656. The CEMS 2023–26 benchmark (EMSR656–898) and the v1.2 blind test
(≥ EMSR899) stay unseen by every model trained on this set. Events Kuro Siwo already has (same EMSR code) are dropped.
Tiles are kept by the paper's own selection: ``is_valid`` and at most 95 % invalid pixels.

**Labels on our grid.**
- ``2`` → flood (1);
- ``0`` → not flood (0);
- ``1`` → not flood, flagged as permanent water;
- ``255`` → unlabelled.

Permanent water is the union of class 1 and JRC occurrence ≥ 80 %, as for Sen1Floods11 and Kuro Siwo. Classes are
read with nearest-neighbour resampling.

**Acquisition.** GEOID-Flood's own post-event Sentinel-1 time, taken from its shard index: the member names
``…_s1grd_post_<YYYYMMDDTHHMMSS>.tif`` give it without downloading 177 GB of imagery. Per AOI, the time shared by
most of its tiles is used, and Planetary Computer's acquisition at that time covering the most of the AOI is taken.

**Chips.** The Kuro Siwo rule on the AOI's labelled footprint: 5.12 km squares, drawn per event by a generator
seeded with the event, at most :data:`N_FLOOD_TILES` with flood ≥ 1 % and :data:`N_DRY_TILES` dry. The cap is fixed in
``reports/v1_2_deviations.md``: the compute budget, and event diversity matter more than tiles per event.

**Integrity.** The dataset revision is pinned (:data:`REVISION`), every file is sha256-checked, and tar members are
extracted only if their names match the label layout.

**Licence.** Compilation, splits and labels are CC BY 4.0. Attribution: "GEOID-Flood (Chiriaco et al., 2026);
contains modified Copernicus Sentinel-1 and Sentinel-2 data (2016–2026) and Copernicus Emergency Management Service
Rapid Mapping products, © European Union". Nothing derived from it is redistributed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import re
import tarfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.data.chipping import AnalysisGrid

__all__ = ["CUTOFF_CODE", "FILES", "REVISION", "GfAoi", "download", "label_classes_on_grid", "load_aois",
           "main", "post_times", "selected_events"]

LOGGER = logging.getLogger(__name__)

REVISION = "868407460bf3db492f50730a57585916baa71dc6"
_REPO = "links-ads/geoid-flood"
_HOST = "huggingface.co"
#: Downloads redirect once, to Hugging Face's CDN (us.aws.cdn.hf.co on 2026-10-13).
_CDN_SUFFIX = ".hf.co"
#: The files used, with their sha256 at :data:`REVISION` (its ``SHA256SUMS``).
FILES = {
    "geoid-flood/tile_catalog.parquet": "c6ec7e3a3c252977069e9fa1613f07a0c01f7a121a0f55e9868f0645aad38022",
    "shard_index.json.gz": "28e9d12cd778fa6cc6577cb0f1b4db34ca39487b56be68ac9792802cbe5b3723",
    "geoid-flood/shards/train/label/train-label-00000-of-00001.tar":
        "50c875def3360e61e2f934887135792023eca875c8b7767794b060ff126188cf",
    "geoid-flood/shards/val/label/val-label-00000-of-00001.tar":
        "2c168542afed97d5250a9a4f12cbcb3e87a15ffd8a71f5ff0e4a6a8fce0a0a12",
    "geoid-flood/shards/test/label/test-label-00000-of-00001.tar":
        "815a828055ead2b852562f4ae2041085dad4ddd0ae0e94dbf5237e691b87f684",
}
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 8 * 1024 * 1024
_LABEL = re.compile(r"(EMSR\d+)-(\d+)/label/\1-\2-(\d+)_label\.tif", re.ASCII)
_S1_POST = re.compile(r"(EMSR\d+)-(\d+)/s1grd/\1-\2-(\d+)_s1grd_post_(\d{8}T\d{6})\.tif", re.ASCII)
#: Activations from this code on are the CEMS benchmark and the blind test: never trained on.
CUTOFF_CODE = 656
#: The paper's tile selection.
MAX_INVALID_FRACTION = 0.95
FLOOD, PERMANENT, BACKGROUND, UNMAPPED = 2, 1, 0, 255
N_FLOOD_TILES = 4
N_DRY_TILES = 1
#: A candidate square needs at least this share of mapped pixels (a tile's footprint can be mostly unmapped).
MIN_MAPPED_FRACTION = 0.5
#: An acquisition is matched to GEOID-Flood's post time within this much.
_TIME_TOLERANCE = timedelta(minutes=5)
GROUP_PREFIX = "GF_"


# ---------------------------------------------------------------------------
# Download: pinned revision, sha256 per file, label members only
# ---------------------------------------------------------------------------
def _url(path: str) -> str:
    return f"https://{_HOST}/datasets/{_REPO}/resolve/{REVISION}/{path}"


def _http_get(url: str) -> bytes:
    from floodfusion.data.http import capped_get, check_https_url

    return capped_get(url, check=lambda u: check_https_url(u, hosts=(_HOST,)), cap=_MAX_FILE_BYTES, retries=3,
                      timeout_s=600.0, redirect_check=lambda u: check_https_url(u, suffixes=(_CDN_SUFFIX,)))


def _extract_labels(blob: bytes, raw_root: Path) -> int:
    """Write the label members of one shard under ``raw_root`` (names validated, sizes capped); return the count."""
    from floodfusion.data.records import atomic_write_bytes

    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        for member in tar:
            if not member.isfile() or not _LABEL.fullmatch(member.name):
                continue
            if member.size > _MAX_MEMBER_BYTES:
                raise OSError(f"{member.name}: larger than {_MAX_MEMBER_BYTES} bytes")
            data = tar.extractfile(member)
            if data is None:   # pragma: no cover - isfile() members always have data
                continue
            atomic_write_bytes(raw_root / member.name, data.read())
            n += 1
    return n


def download(raw_root: Path | str, *, get: Callable[[str], bytes] = _http_get) -> dict[str, int]:
    """Fetch :data:`FILES` at :data:`REVISION`, sha256-verified; extract the labels. Resumable per file."""
    from floodfusion.data.records import atomic_write_bytes

    raw_root = Path(raw_root)
    done: dict[str, int] = {}
    for path, sha in FILES.items():
        marker = raw_root / "downloaded" / (path.replace("/", "__") + ".sha256")
        if marker.is_file() and marker.read_text().strip() == sha:
            done[path] = -1
            continue
        blob = get(_url(path))
        digest = hashlib.sha256(blob).hexdigest()
        if digest != sha:
            raise ValueError(f"{path}: sha256 {digest[:12]}… is not the pinned {sha[:12]}…")
        if path.endswith(".tar"):
            done[path] = _extract_labels(blob, raw_root)
        else:
            atomic_write_bytes(raw_root / path, blob)
            done[path] = 1
        atomic_write_bytes(marker, sha.encode())
    return done


# ---------------------------------------------------------------------------
# Events, AOIs, times
# ---------------------------------------------------------------------------
def _code(event: str) -> int:
    match = re.fullmatch(r"EMSR(\d+)", str(event))
    if match is None:
        raise ValueError(f"not a CEMS rapid-mapping code: {str(event)[:20]!r}")
    return int(match.group(1))


def selected_events(catalog: Any, exclude_codes: Sequence[str] = ()) -> list[str]:
    """Activations before :data:`CUTOFF_CODE` with at least one usable tile, minus ``exclude_codes`` (Kuro Siwo's)."""
    usable = catalog[catalog["is_valid"].astype(bool) & (catalog["invalid_pixel_frac"] <= MAX_INVALID_FRACTION)]
    events = {e for e in usable["event_id"].unique() if _code(e) < CUTOFF_CODE}
    return sorted(events - set(exclude_codes))


def post_times(index: Mapping[str, Any]) -> dict[tuple[str, int], datetime]:
    """``{(event, aoi): post-event Sentinel-1 time}``, the time most of the AOI's tiles carry (ties: the earliest)."""
    per_aoi: dict[tuple[str, int], Counter[str]] = {}
    for shard in index["shards"]:
        if "/s1grd/" not in shard["path"]:
            continue
        for name in shard["members"]:
            match = _S1_POST.fullmatch(name)
            if match:
                per_aoi.setdefault((match.group(1), int(match.group(2))), Counter())[match.group(4)] += 1
    out = {}
    for key, counts in per_aoi.items():
        top = max(counts.values())
        stamp = min(t for t, c in counts.items() if c == top)
        out[key] = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    return out


@dataclass(frozen=True)
class GfAoi:
    """One GEOID-Flood AOI: its usable label tiles (path, lon/lat footprint) and post-event time."""

    event: str
    aoi: int
    tiles: tuple[tuple[str, Any], ...]
    post: datetime

    @property
    def name(self) -> str:
        return f"{GROUP_PREFIX}{self.event}_{self.aoi:02d}"

    @property
    def region(self) -> Any:
        from shapely.ops import unary_union

        return unary_union([g for _, g in self.tiles])


def load_aois(raw_root: Path, events: Sequence[str]) -> list[GfAoi]:
    """The AOIs of ``events`` that have both usable label tiles on disk and a post-event time."""
    import pandas as pd
    import shapely

    catalog = pd.read_parquet(raw_root / "geoid-flood" / "tile_catalog.parquet")
    index = json.loads(gzip.decompress((raw_root / "shard_index.json.gz").read_bytes()))
    times = post_times(index)
    usable = catalog[catalog["is_valid"].astype(bool) & (catalog["invalid_pixel_frac"] <= MAX_INVALID_FRACTION)
                     & catalog["event_id"].isin(events)]
    out = []
    for (event, aoi), rows in usable.groupby(["event_id", "tile_id"]):
        aoi = int(aoi)
        when = times.get((event, aoi))
        tiles = []
        for _, row in rows.sort_values("bbox_id", key=lambda c: c.astype(int)).iterrows():
            path = raw_root / f"{event}-{aoi}" / "label" / f"{event}-{aoi}-{int(row['bbox_id'])}_label.tif"
            if path.is_file():
                tiles.append((str(path), shapely.from_wkb(row["geometry"])))
        if when is None or not tiles:
            LOGGER.warning("%s-%d: %s", event, aoi, "no post-event time" if when is None else "no label on disk")
            continue
        out.append(GfAoi(event, aoi, tuple(tiles), when))
    return out


# ---------------------------------------------------------------------------
# Labels on a grid
# ---------------------------------------------------------------------------
def label_classes_on_grid(aoi: GfAoi, grid: AnalysisGrid) -> np.ndarray:
    """GEOID-Flood's classes (uint8, 255 = unlabelled) on ``grid``: the AOI's tiles, nearest, first valid wins."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    from floodfusion.data.chipping import reproject_to_grid

    out = np.full((grid.height, grid.width), UNMAPPED, dtype=np.uint8)
    footprint = box(*transform_bounds(CRS.from_epsg(grid.epsg), CRS.from_epsg(4326), *grid.bounds))
    for path, geometry in aoi.tiles:
        if not geometry.intersects(footprint):
            continue
        layer = reproject_to_grid(path, grid, kind="categorical", nodata=UNMAPPED, src_nodata=UNMAPPED, dtype="uint8")
        gap = (out == UNMAPPED) & (layer != UNMAPPED)
        out[gap] = layer[gap]
    return out


def _label(classes: np.ndarray) -> np.ndarray:
    """1 flood, 0 background or permanent water, NaN unmapped."""
    return np.where(classes == FLOOD, 1.0, np.where((classes == BACKGROUND) | (classes == PERMANENT), 0.0, np.nan))


def _reference(aoi: GfAoi, grid: AnalysisGrid, jrc: Sequence[Mapping[str, Any]],
               worldcover: Sequence[Mapping[str, Any]]) -> Any:
    from floodfusion.data.assembly import GridReference
    from floodfusion.data.sources import read_jrc_occurrence, read_worldcover
    from floodfusion.masks import permanent_water_from_occurrence

    classes = label_classes_on_grid(aoi, grid)
    label = _label(classes)
    permanent = classes == PERMANENT
    if jrc:
        permanent |= permanent_water_from_occurrence(np.nan_to_num(read_jrc_occurrence(jrc, grid), nan=0.0))
    label = np.where(permanent & (label == 1), 0.0, label)
    landcover = read_worldcover(worldcover, grid) if worldcover else np.zeros((grid.height, grid.width), np.uint8)
    return GridReference(label=label, inside_aoi=np.isfinite(label), permanent=permanent, landcover=landcover)


def _source(aoi: GfAoi, acq: Any, group: str) -> Any:
    from floodfusion.data.kurosiwo import LabelSource

    return LabelSource(
        name=aoi.name,
        label=lambda grid: _label(label_classes_on_grid(aoi, grid)),
        reference=lambda grid, jrc, worldcover: _reference(aoi, grid, jrc, worldcover),
        meta=lambda tile: {"geoid_flood": {"event": aoi.event, "aoi": aoi.aoi, "group": group, "revision": REVISION,
                                           "post_acquisition": acq.when.isoformat(),
                                           "geoid_post_time": aoi.post.isoformat(),
                                           "flood_fraction": tile.flood_fraction}},
        generator="floodfusion.data.geoid (labels: GEOID-Flood rasters; inputs: Planetary Computer)",
    )


# ---------------------------------------------------------------------------
# Planning and ingest
# ---------------------------------------------------------------------------
_PLAN_RES_M = 80.0


def _shares(aoi: GfAoi, tile: Any) -> tuple[float, float]:
    """``(flood share of the mapped pixels, mapped share)`` of one candidate square, on an 80 m grid (planning)."""
    from floodfusion.data.kurosiwo import CHIP_PX

    epsg, x0, y0 = tile.utm                       # south-west corner of a CHIP_PX x 10 m square
    n = int(round(CHIP_PX * 10.0 / _PLAN_RES_M))
    grid = AnalysisGrid(epsg=int(epsg), transform=(_PLAN_RES_M, 0.0, x0, 0.0, -_PLAN_RES_M, y0 + n * _PLAN_RES_M),
                        width=n, height=n)
    classes = label_classes_on_grid(aoi, grid)
    mapped = classes != UNMAPPED
    return (float((classes == FLOOD).sum() / mapped.sum()) if mapped.any() else 0.0), float(mapped.mean())


@dataclass(frozen=True)
class _Plan:
    aoi: GfAoi
    acquisition: Any
    coverage: float
    tiles: tuple[Any, ...]
    shift_tiles: tuple[Any, ...]


def plan_event(event: str, aois: Sequence[GfAoi], client: Any) -> list[_Plan]:
    """Per AOI, the Planetary Computer acquisition at GEOID-Flood's post time, and the event's sampled squares."""
    from dataclasses import replace

    from floodfusion.data.kurosiwo import (
        MIN_AOI_COVERAGE,
        candidate_tiles,
        sample_tiles,
        shift_sample,
    )
    from floodfusion.data.passes import acquisitions, best_covering

    plans = []
    for aoi in aois:
        region = aoi.region
        chosen = best_covering(acquisitions(client, region, aoi.post - _TIME_TOLERANCE, aoi.post + _TIME_TOLERANCE),
                               region)
        if chosen is None or chosen[1] < MIN_AOI_COVERAGE:
            LOGGER.warning("%s: no acquisition at %s covers >= %.0f %% of the AOI", aoi.name, aoi.post.isoformat(),
                           100 * MIN_AOI_COVERAGE)
            continue
        acq, share = chosen
        squares = []
        for t in candidate_tiles(region.intersection(acq.footprint), None, aoi=aoi.name):
            flood, mapped = _shares(aoi, t)
            if mapped >= MIN_MAPPED_FRACTION:
                squares.append(replace(t, flood_fraction=flood))
        plans.append((aoi, acq, share, squares))
    picked = {t.tile_id for t in sample_tiles([t for *_, ts in plans for t in ts], event, n_flood=N_FLOOD_TILES,
                                              n_dry=N_DRY_TILES)}
    out = []
    for aoi, acq, share, candidates in plans:
        tiles = tuple(t for t in candidates if t.tile_id in picked)
        if tiles:
            out.append(_Plan(aoi, acq, share, tiles, shift_sample(tiles, candidates, aoi.name)))
    return out


def ingest_geoid(raw_root: Path | str, out_root: Path | str, cfg: Any, *, exclude_codes: Sequence[str] = (),
                 events: Sequence[str] | None = None, sources: Any = None, margin_px: int | None = None,
                 workers: int = 4) -> dict[str, Any]:
    """Plan and ingest every selected event (or those of ``events``); write ``ingest_report.json``. Resumable."""
    import pandas as pd

    from floodfusion.data.archive import PC_STAC_URL, IngestSources
    from floodfusion.data.kurosiwo import DEFAULT_MARGIN_PX, _ingest_aoi
    from floodfusion.data.records import atomic_write_bytes

    raw_root, out_root = Path(raw_root), Path(out_root)
    catalog = pd.read_parquet(raw_root / "geoid-flood" / "tile_catalog.parquet")
    names = selected_events(catalog, exclude_codes) if events is None else sorted(events)
    by_event: dict[str, list[GfAoi]] = {}
    for aoi in load_aois(raw_root, names):
        by_event.setdefault(aoi.event, []).append(aoi)
    sources = sources or IngestSources.open(PC_STAC_URL)
    margin = DEFAULT_MARGIN_PX if margin_px is None else margin_px
    report: dict[str, Any] = {"revision": REVISION, "events": {}, "excluded": {}}
    for name in names:
        if name not in by_event:
            report["excluded"][name] = "no AOI with labels and a post-event time"
            continue
        plans = plan_event(name, by_event[name], sources.client)
        if not plans:
            report["excluded"][name] = "no covering acquisition or no square inside the AOI"
            continue
        group = f"{GROUP_PREFIX}{name}"   # provisional: the pooled set regroups jointly with Kuro Siwo
        status: dict[str, Any] = {}
        for plan in plans:
            LOGGER.info("%s: %s o%03d (%.0f %% of the AOI), %d tiles", plan.aoi.name,
                        plan.acquisition.when.isoformat(), plan.acquisition.relative_orbit, 100 * plan.coverage,
                        len(plan.tiles))
            status[plan.aoi.name] = _ingest_aoi(_source(plan.aoi, plan.acquisition, group), plan.acquisition,
                                                plan.tiles, group, out_root, cfg, sources, margin, workers,
                                                shift_tiles=plan.shift_tiles)
        report["events"][name] = status
        out_root.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(out_root / "ingest_report.json", json.dumps(report, indent=2, default=str).encode())
    outcomes = [o for ev in report["events"].values() for st in ev.values() for k, o in st.items()
                if k != "event_shift" and isinstance(o, str)]
    report["summary"] = {k: sum(o.startswith(k) for o in outcomes) for k in ("ok", "done-before", "dropped", "failed")}
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_root / "ingest_report.json", json.dumps(report, indent=2, default=str).encode("utf-8"))
    return report


def kurosiwo_codes(events_file: Path) -> list[str]:
    """Every EMSR code Kuro Siwo annotates (``configs/datasets/kurosiwo_events.yaml``, dated or not): dropped here."""
    import yaml

    names = yaml.safe_load(Path(events_file).read_text(encoding="utf-8"))["events"]
    return sorted({m.group(1) for name in names if (m := re.match(r"(EMSR\d+)_", name))})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GEOID-Flood labels, re-ingested on Planetary Computer inputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--raw", type=Path, default=Path("data/geoid_raw"))
    ing = sub.add_parser("ingest")
    ing.add_argument("--raw", type=Path, default=Path("data/geoid_raw"))
    ing.add_argument("--out", type=Path, default=Path("data/geoid"))
    ing.add_argument("--config", type=Path, default=Path("configs/base.yaml"))   # as Kuro Siwo's ingest
    ing.add_argument("--overlay", type=Path, action="append", default=[])
    ing.add_argument("--kurosiwo-events", type=Path, default=Path("configs/datasets/kurosiwo_events.yaml"))
    ing.add_argument("--events", nargs="*", default=None)
    ing.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "download":
        LOGGER.info("%s", download(args.raw))
        return 0
    from floodfusion.config import load_config

    cfg = load_config(args.config, overlays=args.overlay)
    report = ingest_geoid(args.raw, args.out, cfg, exclude_codes=kurosiwo_codes(args.kurosiwo_events),
                          events=args.events, workers=args.workers)
    LOGGER.info("summary: %s; excluded: %d events", report["summary"], len(report["excluded"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
