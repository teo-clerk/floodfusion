r"""Copernicus GFM (Global Flood Monitoring) layers on an event stack's grid: v1.2's operational comparator.

GFM, part of the Copernicus Emergency Management Service, maps floods in every Sentinel-1 IW GRD scene. Three
algorithms run (DLR, LIST, TU Wien), and a pixel is flooded when the majority says so. EODC serves the archive as a
public STAC collection, with no account: :data:`STAC_SEARCH_URL`, collection ``GFM``.

- **Items.** One item per (Sentinel-1 frame, Equi7Grid T3 tile): 20 m ``uint8`` COGs with nodata 255.
- **Matching.** An item's id carries the start time of its frame. For example,
  ``ENSEMBLE_FLOOD_20230408T155612_VV_EU020M_E057N024T3`` belongs to frame ``S1A_IW_GRDH_1SDV_20230408T155612_…``.
  That is how an item is matched to the frames our stack was built from: by time, never by footprint alone.
- **Reading.** Each layer is warped once onto the stack's 10 m grid with nearest-neighbour resampling, since flood
  classes and likelihood values must not be interpolated. Only the source blocks the grid needs are read.
- **Ensemble likelihood.** The mean of the members' likelihoods, rescaled from 0 (low) to 100 (high) flood
  likelihood (Krullikowski et al., 2023, IEEE JSTARS).

GFM is a comparator here, never a model input (``reports/v1_2_deviations.md``): its members change in 2027.

Products: © European Union, Copernicus Emergency Management Service (GFM).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.data.chipping import AnalysisGrid

__all__ = ["COLLECTION", "LAYERS", "NODATA", "STAC_SEARCH_URL", "build_sidecar", "frame_start", "main",
           "match_frames", "read_layers", "read_sidecar", "search_items"]

LOGGER = logging.getLogger(__name__)

STAC_SEARCH_URL = "https://stac.eodc.eu/api/v1/search"
COLLECTION = "GFM"
#: Hosts each request may go to: the catalogue, and the COG store its assets point at.
STAC_HOSTS = ("stac.eodc.eu",)
DATA_HOSTS = ("data.eodc.eu",)
#: (sidecar band, STAC asset key), in band order.
LAYERS: tuple[tuple[str, str], ...] = (
    ("ensemble_flood", "ensemble_flood_extent"),
    ("ensemble_likelihood", "ensemble_likelihood"),
    ("exclusion", "exclusion_mask"),
    ("reference_water", "reference_water_mask"),
    ("advisory", "advisory_flags"),
    ("tuw_flood", "tuw_flood_extent"),
    ("dlr_flood", "dlr_flood_extent"),
    ("list_flood", "list_flood_extent"),
)
NODATA = 255
SIDECAR = "gfm.tif"
RECORD = "gfm.json"
#: A frame's start time is matched within this window (STAC datetimes are the frame's own times).
_SEARCH_PAD = timedelta(minutes=2)
_MAX_SEARCH_BYTES = 8 * 1024 * 1024
_MAX_PAGES = 10
_START = re.compile(r"\d{8}T\d{6}")


def frame_start(scene_id: str) -> str:
    """``YYYYMMDDTHHMMSS``: the start time in a Sentinel-1 product id (its fifth field)."""
    fields = str(scene_id).split("_")
    if len(fields) < 5 or not _START.fullmatch(fields[4]):
        raise ValueError(f"not a Sentinel-1 product id: {str(scene_id)[:80]!r}")
    return fields[4]


def _default_get(url: str, params: Mapping[str, str] | None = None) -> bytes:
    from floodfusion.data.http import capped_get, check_https_url

    return capped_get(url, check=lambda u: check_https_url(u, hosts=STAC_HOSTS), cap=_MAX_SEARCH_BYTES,
                      params=params, retries=3)


def search_items(bbox: Sequence[float], starts: Sequence[str], *,
                 get: Callable[..., bytes] = _default_get) -> list[dict[str, Any]]:
    """Every GFM item over ``bbox`` (lon/lat) acquired within :data:`_SEARCH_PAD` of the frame starts."""
    times = [datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=UTC) for s in starts]
    if not times:
        raise ValueError("no frame to search for")
    window = f"{min(times) - _SEARCH_PAD:%Y-%m-%dT%H:%M:%SZ}/{max(times) + _SEARCH_PAD:%Y-%m-%dT%H:%M:%SZ}"
    params = {"collections": COLLECTION, "bbox": ",".join(f"{v:.6f}" for v in bbox), "datetime": window,
              "limit": "100"}
    items: list[dict[str, Any]] = []
    url: str | None = STAC_SEARCH_URL
    for _ in range(_MAX_PAGES):
        page = json.loads(get(url, params=params))
        items += list(page.get("features", []))
        url = next((link.get("href") for link in page.get("links", []) if link.get("rel") == "next"), None)
        if url is None:
            return items
        params = None   # a "next" link carries its own query
    raise RuntimeError(f"more than {_MAX_PAGES} pages of GFM items for one window: {window}")


def match_frames(items: Sequence[Mapping[str, Any]], starts: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """``{start: [items of that frame]}``, one entry per frame start; items of other frames are dropped."""
    return {s: [dict(i) for i in items if f"_{s}_" in str(i.get("id", ""))] for s in starts}


def _asset_href(item: Mapping[str, Any], key: str) -> str:
    from floodfusion.data.http import check_https_url

    try:
        href = item["assets"][key]["href"]
    except KeyError as exc:
        raise KeyError(f"GFM item {item.get('id')!r} has no asset {key!r}") from exc
    return check_https_url(href, hosts=DATA_HOSTS)


def _remote(href: str) -> str:
    return "/vsicurl/" + href


def read_layers(items: Sequence[Mapping[str, Any]], grid: AnalysisGrid, *,
                source: Callable[[str], str] = _remote) -> np.ndarray:
    """``(len(LAYERS), H, W)`` uint8 on ``grid``: each layer mosaicked over ``items`` (first valid value wins)."""
    import rasterio

    from floodfusion.data.archive import GDAL_ENV
    from floodfusion.data.chipping import reproject_to_grid

    out = np.full((len(LAYERS), grid.height, grid.width), NODATA, dtype=np.uint8)
    env = {**GDAL_ENV, "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env):
        for band, (_, key) in enumerate(LAYERS):
            for item in items:
                gap = out[band] == NODATA
                if not gap.any():
                    break
                layer = reproject_to_grid(source(_asset_href(item, key)), grid, kind="categorical",
                                          nodata=NODATA, src_nodata=NODATA, dtype="uint8")
                fill = gap & (layer != NODATA)
                out[band][fill] = layer[fill]
    return out


def _grid(event_dir: Path) -> AnalysisGrid:
    meta = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    return AnalysisGrid(epsg=int(meta["epsg"]), transform=tuple(float(v) for v in meta["transform"]),
                        width=int(meta["width"]), height=int(meta["height"]))


def _lonlat_bbox(grid: AnalysisGrid) -> tuple[float, float, float, float]:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    return tuple(transform_bounds(CRS.from_epsg(grid.epsg), CRS.from_epsg(4326), *grid.bounds, densify_pts=21))


def _item_record(item: Mapping[str, Any]) -> dict[str, Any]:
    props = item.get("properties", {})
    return {"id": item.get("id"), "datetime": props.get("datetime"), "version": props.get("processing:version"),
            "processed": props.get("processing:datetime"), "equi7_tile": props.get("Equi7Tile"),
            "members": props.get("flood_members"), "assets": {key: _asset_href(item, key) for _, key in LAYERS}}


def _write(path: Path, layers: np.ndarray, grid: AnalysisGrid) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    profile = {"driver": "GTiff", "dtype": "uint8", "count": len(LAYERS), "height": grid.height,
               "width": grid.width, "crs": CRS.from_epsg(grid.epsg), "transform": Affine(*grid.transform),
               "nodata": NODATA, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(layers)
        for band, (name, _) in enumerate(LAYERS, start=1):
            dst.set_band_description(band, name)
    tmp.replace(path)


def build_sidecar(event_dir: Path, *, search: Callable[..., list[dict[str, Any]]] = search_items,
                  source: Callable[[str], str] = _remote, overwrite: bool = False) -> dict[str, Any]:
    """Write ``gfm.tif`` (if every post-event frame has a GFM product) and ``gfm.json`` beside one stack."""
    from floodfusion.data.records import atomic_write_bytes

    record_path = event_dir / RECORD
    if record_path.is_file() and not overwrite:
        return json.loads(record_path.read_text(encoding="utf-8"))
    scenes = json.loads((event_dir / "scenes.json").read_text(encoding="utf-8"))[0]
    frames = list(scenes.get("s1_post_frames") or [scenes["s1_post_id"]])
    starts = sorted({frame_start(f) for f in frames})
    grid = _grid(event_dir)
    matched = match_frames(search(_lonlat_bbox(grid), starts), starts)
    missing = [s for s, items in matched.items() if not items]
    record: dict[str, Any] = {"frames": frames, "starts": starts, "collection": COLLECTION,
                              "layers": [name for name, _ in LAYERS]}
    if missing:
        record.update(status=f"no GFM product for frame start(s) {', '.join(missing)}", items=[])
    else:
        items = [i for s in starts for i in matched[s]]
        layers = read_layers(items, grid, source=source)
        _write(event_dir / SIDECAR, layers, grid)
        record.update(status="ok", items=[_item_record(i) for i in items],
                      versions=sorted({str(i.get("properties", {}).get("processing:version")) for i in items}),
                      grid_coverage=float(np.mean(layers[0] != NODATA)),
                      layer_sha256={name: hashlib.sha256(layers[b].tobytes()).hexdigest()
                                    for b, (name, _) in enumerate(LAYERS)})
    atomic_write_bytes(record_path, json.dumps(record, indent=1).encode("utf-8"))
    return record


def read_sidecar(event_dir: Path) -> dict[str, np.ndarray] | None:
    """``{layer: uint8 array}`` of a stack's GFM sidecar, or ``None`` if it has none (nodata stays 255)."""
    import rasterio

    path = event_dir / SIDECAR
    if not path.is_file():
        return None
    with rasterio.open(path) as src:
        data = src.read()
    return {name: data[b] for b, (name, _) in enumerate(LAYERS)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copernicus GFM layers beside every event stack of a dataset root.")
    parser.add_argument("--root", type=Path, nargs="+", required=True,
                        help="dataset roots, each holding events/*/stack.tif")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from concurrent.futures import ThreadPoolExecutor

    from floodfusion.data.http import redact

    dirs = sorted(d for root in args.root for d in (root / "events").glob("*") if (d / "scenes.json").is_file())

    def one(event_dir: Path) -> str:
        try:
            return str(build_sidecar(event_dir, overwrite=args.overwrite)["status"])
        except Exception as exc:  # noqa: BLE001 - one stack's failure is reported, the sweep resumes on re-run
            LOGGER.error("%s: %s", event_dir.name, redact(str(exc))[:300])
            return "error"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        statuses = list(pool.map(one, dirs))
    ok = sum(s == "ok" for s in statuses)
    errors = sum(s == "error" for s in statuses)
    LOGGER.info("%d stacks: %d with GFM, %d without, %d errors (re-run to retry)", len(dirs), ok,
                len(dirs) - ok - errors, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
