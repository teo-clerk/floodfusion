r"""The seasonal reference: what the backscatter normally is at this time of year.

The ingest's pre-event reference is the **latest** same-orbit pass up to 60 days before the
flood image (:func:`floodfusion.data.stac.search_matched_pairs`): 5-24 days earlier on the
Sen1Floods11 events, and during a monsoon flood that pass can already be flooded. A change
term against it then misses water that was there "before" — whether that water was seasonal or
an earlier stage of the same flood.

This module adds a second reference beside each event stack: the median, in linear power, of
every same-orbit acquisition within :data:`WINDOW_DAYS` days of the same date **one year
earlier** (at least :data:`MIN_SCENES`; otherwise none is written and the record says why). It
is the seasonal expectation the Copernicus Global Flood Monitoring service compares against,
reduced to one window. It is written as ``seasonal.tif`` (VV, VH in dB, float32, NaN = no data)
on the stack's grid, with ``seasonal.json``; the stack itself is never modified.

It serves two purposes, fixed before any v1.1 number (``reports/protocol_v1_1.md``): the label
diagnostic's decomposition (water already there a year earlier vs. a flood that began before the
recent reference) and the physics baseline's change term (``baseline.reference: seasonal``).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MIN_SCENES", "WINDOW_DAYS", "anniversary_window", "main", "read_seasonal", "seasonal_reference"]

LOGGER = logging.getLogger(__name__)

WINDOW_DAYS = 30
MIN_SCENES = 3
_YEAR = timedelta(days=365)


def anniversary_window(post: datetime) -> tuple[datetime, datetime]:
    """``WINDOW_DAYS`` either side of the same date one year before ``post``."""
    centre = post - _YEAR
    return centre - timedelta(days=WINDOW_DAYS), centre + timedelta(days=WINDOW_DAYS)


def _grid_polygon(grid: Any) -> Any:
    from pyproj import Transformer
    from shapely.geometry import box

    to = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    x0, y0, x1, y1 = grid.bounds
    lon, lat = to.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return box(min(lon), min(lat), max(lon), max(lat))


def seasonal_reference(event_dir: Path, client: Any, *, overwrite: bool = False) -> dict[str, Any]:
    """Write ``seasonal.tif`` / ``seasonal.json`` beside ``event_dir/stack.tif``; return the record."""
    import rasterio

    from floodfusion.data.archive import GDAL_ENV, items_by_id
    from floodfusion.data.chipping import read_event_stack
    from floodfusion.data.passes import acquisitions
    from floodfusion.data.preprocessing import db_to_linear, linear_to_db
    from floodfusion.data.records import atomic_write_bytes
    from floodfusion.data.sources import read_s1_rtc

    record_path = event_dir / "seasonal.json"
    if record_path.is_file() and not overwrite:
        return json.loads(record_path.read_text(encoding="utf-8"))
    meta = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    post = datetime.fromisoformat(str(meta["s1_post_datetime"]))
    orbit, state = int(meta["relative_orbit"]), str(meta["orbit_state"]).lower()
    _, grid, _ = read_event_stack(event_dir / "stack.tif")
    start, end = anniversary_window(post)
    candidates = [a for a in acquisitions(client, _grid_polygon(grid), start, end)
                  if a.relative_orbit == orbit and a.orbit_state == state]
    record: dict[str, Any] = {"window": [start.isoformat(), end.isoformat()], "relative_orbit": orbit,
                              "orbit_state": state, "acquisitions": [a.when.isoformat() for a in candidates]}
    if len(candidates) < MIN_SCENES:
        record.update(status=f"insufficient: {len(candidates)} < {MIN_SCENES} same-orbit acquisitions", n_scenes=0)
        atomic_write_bytes(record_path, json.dumps(record, indent=2).encode("utf-8"))
        return record
    stacks: dict[str, list[np.ndarray]] = {"VV": [], "VH": []}
    skipped: list[str] = []
    with rasterio.Env(**GDAL_ENV):
        for acq in candidates:
            items = items_by_id(client, "sentinel-1-rtc", acq.frame_ids)
            try:
                vv, _ = read_s1_rtc(items, grid, "VV")
            except ValueError:   # the frame's footprint touches the grid but holds no valid pixel over it
                skipped.append(acq.when.isoformat())
                continue
            stacks["VV"].append(db_to_linear(vv))
            try:
                vh, _ = read_s1_rtc(items, grid, "VH")
                stacks["VH"].append(db_to_linear(vh))
            except (KeyError, ValueError):   # single polarisation (1SSV) or no VH data: VV is what counts
                pass
    record["skipped_acquisitions"] = skipped
    if len(stacks["VV"]) < MIN_SCENES:
        record.update(status=f"insufficient: {len(stacks['VV'])} < {MIN_SCENES} acquisitions with data over the grid",
                      n_scenes=0)
        atomic_write_bytes(record_path, json.dumps(record, indent=2).encode("utf-8"))
        return record
    layers = []
    for pol in ("VV", "VH"):
        if len(stacks[pol]) < MIN_SCENES:   # VH is optional; the change term uses VV
            layers.append(np.full((grid.height, grid.width), np.nan, dtype=np.float32))
            continue
        with np.errstate(all="ignore"):
            median = np.nanmedian(np.stack(stacks[pol]), axis=0)
        layers.append(np.where(median > 0, linear_to_db(np.where(median > 0, median, 1.0)), np.nan).astype(np.float32))
    record["n_vh_scenes"] = len(stacks["VH"])
    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width, "count": 2, "dtype": "float32",
               "crs": f"EPSG:{grid.epsg}", "transform": grid.affine, "nodata": float("nan"), "compress": "deflate",
               "tiled": True}
    tmp = event_dir / ".seasonal.tmp.tif"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(np.stack(layers))
        dst.set_band_description(1, "seasonal_vv_db")
        dst.set_band_description(2, "seasonal_vh_db")
    tmp.replace(event_dir / "seasonal.tif")
    record.update(status="ok", n_scenes=len(stacks["VV"]))
    atomic_write_bytes(record_path, json.dumps(record, indent=2).encode("utf-8"))
    return record


def read_seasonal(event_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """``(vv_db, vh_db)`` of an event's seasonal reference, or ``None`` if it has none."""
    import rasterio

    path = event_dir / "seasonal.tif"
    if not path.is_file():
        return None
    with rasterio.open(path) as src:
        vv, vh = src.read()
    return vv.astype(np.float64), vh.astype(np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seasonal Sentinel-1 reference beside every event stack.")
    parser.add_argument("--root", type=Path, required=True, help="a dataset root holding events/*/stack.tif")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from concurrent.futures import ThreadPoolExecutor

    from floodfusion.data.archive import IngestSources
    from floodfusion.data.stac import redact

    client = IngestSources.open().client
    dirs = sorted(d for d in (args.root / "events").glob("*") if (d / "event.json").is_file())

    def run(d: Path) -> tuple[str, str]:
        try:
            return d.name, seasonal_reference(d, client, overwrite=args.overwrite)["status"]
        except Exception as exc:  # noqa: BLE001 - one event's failure must not stop the dataset
            return d.name, f"failed: {type(exc).__name__}: {redact(str(exc))[:200]}"

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, status in pool.map(run, dirs):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            LOGGER.info("%s: %s", name, status)
    LOGGER.info("seasonal references: %s", counts)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
