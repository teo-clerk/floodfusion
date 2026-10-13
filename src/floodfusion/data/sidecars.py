r"""Per-stack sidecar layers read beside the 16-band stack: the context of v1.2's models.

The stack's contract (:mod:`floodfusion.data.encoding`) stays untouched. Each sidecar is its own raster beside
``stack.tif``, on the same grid. A model asks for sidecars through its **architecture**
(:data:`SIDECARS_BY_ARCHITECTURE`), never through new config fields: new fields would change the hash of every
configuration, and with it the protocol records of runs already made.

========== ============== ==== ===================================================================================
sidecar    file           ch   content
========== ============== ==== ===================================================================================
seasonal   seasonal.tif   3    same-orbit median γ⁰ VV, VH of the same season a year earlier
                                (:mod:`floodfusion.data.seasonal`), and VV post − VV seasonal (dB)
aef        aef.tif        8    AlphaEarth embedding of the year before the event, first 8 principal components
dem        dem.tif        1    Copernicus GLO-30 height (m, EGM2008): TerraMind's native DEM modality
========== ============== ==== ===================================================================================

Normalisation uses fixed statistics only, never the chip's own:
- seasonal VV and VH use the dataset's pre-event SAR statistics, and the change uses its change statistics;
- AEF components are divided by their stored scale;
- the DEM stays in metres (the TerraMind wrapper applies its pre-training statistics).

Where a stack has no sidecar, or the window holds no valid value, ``available`` is False for that sidecar. The
values are then zero and meaningless, and the model substitutes a learned embedding, as it does for missing optical.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SIDECARS", "SIDECARS_BY_ARCHITECTURE", "Sidecar", "build_dem", "channels", "link_from",
           "normalise_sidecar", "read_sidecar_window", "sidecars_for"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sidecar:
    name: str
    filename: str
    raw_bands: int      # bands stored in the file
    n_channels: int     # channels handed to the model


SIDECARS: dict[str, Sidecar] = {s.name: s for s in (Sidecar("seasonal", "seasonal.tif", 2, 3),
                                                       Sidecar("aef", "aef.tif", 8, 8),
                                                       Sidecar("dem", "dem.tif", 1, 1))}
#: Which sidecars an architecture reads, in channel order.
SIDECARS_BY_ARCHITECTURE: dict[str, tuple[str, ...]] = {
    "fusion_crossattn_ctx_seasonal": ("seasonal",),
    "fusion_crossattn_ctx_aef": ("aef",),
    "fusion_crossattn_ctx": ("seasonal", "aef"),
    "terramind_base_fair": ("dem",),
    "terramind_base_fair_frozen": ("dem",),
    "terramind_small_fair": ("dem",),
}
#: AEF principal components are stored as int8 of value x scale (configs/datasets/aef_pca_v12.json fixes the scale).
AEF_STORE_SCALE = 127.0


def sidecars_for(architecture: str) -> tuple[str, ...]:
    return SIDECARS_BY_ARCHITECTURE.get(architecture, ())


def channels(names: Sequence[str]) -> int:
    return sum(SIDECARS[n].n_channels for n in names)


def read_sidecar_window(handle: Any, window: tuple[int, int, int, int], context_px: int,
                        n_bands: int) -> np.ndarray:
    """``(bands, h + 2c, w + 2c)`` float32 of a sidecar's raw window; NaN outside the raster or at nodata."""
    from rasterio.windows import Window

    col, row, width, height = window
    c = context_px
    data = handle.read(indexes=list(range(1, n_bands + 1)), window=Window(col - c, row - c, width + 2 * c,
                                                                           height + 2 * c),
                       boundless=True, masked=True, out_dtype="float32")
    return np.ma.filled(data.astype(np.float32), np.nan)


def normalise_sidecar(name: str, raw: np.ndarray, *, vv_post_db: np.ndarray | None, norm: Any) -> np.ndarray:
    """Fixed-statistics normalisation of one sidecar window (NaN kept: the caller turns it into availability)."""
    if name == "seasonal":
        vv, vh = raw[0], raw[1]
        change = (vv_post_db - vv) if vv_post_db is not None else np.full_like(vv, np.nan)
        mean, std = norm.sar_mean_db, norm.sar_std_db
        return np.stack([(vv - mean[2]) / std[2], (vh - mean[3]) / std[3], (change - mean[4]) / std[4]])
    if name == "aef":
        return raw[: SIDECARS["aef"].n_channels] / AEF_STORE_SCALE
    if name == "dem":
        return raw[:1]
    raise KeyError(f"unknown sidecar {name!r}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _grid(event_dir: Path) -> Any:
    from floodfusion.data.chipping import AnalysisGrid

    meta = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    transform = meta["transform"] if isinstance(meta["transform"], list) else json.loads(meta["transform"])
    return AnalysisGrid(epsg=int(meta["epsg"]), transform=tuple(float(v) for v in transform),
                        width=int(meta["width"]), height=int(meta["height"])), meta


def _write_float(path: Path, data: np.ndarray, grid: Any, descriptions: Sequence[str]) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    tmp = path.with_suffix(".tmp.tif")
    profile = {"driver": "GTiff", "dtype": "float32", "count": data.shape[0], "height": grid.height,
               "width": grid.width, "crs": CRS.from_epsg(grid.epsg), "transform": Affine(*grid.transform),
               "nodata": float("nan"), "compress": "deflate", "predictor": 3, "tiled": True,
               "blockxsize": 256, "blockysize": 256}
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def build_dem(event_dir: Path, client: Any, *, overwrite: bool = False) -> str:
    """``dem.tif``: the stack's own GLO-30 tiles (``event.json`` ``dem_ids``) on its grid, bilinear, metres."""
    import rasterio

    from floodfusion.data.archive import GDAL_ENV, items_by_id
    from floodfusion.data.sources import read_dem

    out = event_dir / SIDECARS["dem"].filename
    if out.is_file() and not overwrite:
        return "done-before"
    grid, meta = _grid(event_dir)
    ids = meta.get("dem_ids")
    ids = json.loads(ids.replace("'", '"')) if isinstance(ids, str) else list(ids or [])
    if not ids:
        return "no dem_ids"
    with rasterio.Env(**GDAL_ENV):
        height = read_dem(items_by_id(client, "cop-dem-glo-30", ids), grid)
    _write_float(out, height[None], grid, ["glo30_height_m"])
    return "ok"


def _same_grid(a: Path, b: Path) -> bool:
    keys = ("epsg", "transform", "width", "height")
    ma, mb = (json.loads((d / "event.json").read_text(encoding="utf-8")) for d in (a, b))
    return all(str(ma.get(k)) == str(mb.get(k)) for k in keys)


def link_from(old_root: Path, new_root: Path, files: Sequence[str]) -> dict[str, int]:
    """Hard-link ``files`` of every old stack into the rebuilt stack of the same name, if both grids are identical.

    For layers that do not depend on what a rebuild changed (the seasonal reference and GFM do not read the DEM):
    no network, and no disk. A grid mismatch, or a file already there, is left alone and counted.
    """
    import os

    counts = {"linked": 0, "present": 0, "grid_differs": 0, "absent": 0}
    for new in sorted((new_root / "events").glob("*")):
        old = old_root / "events" / new.name
        if not (old / "event.json").is_file() or not (new / "event.json").is_file():
            counts["absent"] += 1
            continue
        if not _same_grid(old, new):
            counts["grid_differs"] += 1
            continue
        for name in files:
            if (new / name).exists():
                counts["present"] += 1
            elif (old / name).is_file():
                os.link(old / name, new / name)
                counts["linked"] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    from floodfusion.data.archive import IngestSources
    from floodfusion.data.http import redact

    parser = argparse.ArgumentParser(description="Sidecar layers beside every event stack of dataset roots.")
    parser.add_argument("layer", choices=("dem", "link"))
    parser.add_argument("--from-root", type=Path, nargs="*", default=[],
                        help="link: the old roots, in the order of --root")
    parser.add_argument("--files", nargs="*", default=["seasonal.tif", "seasonal.json", "gfm.tif", "gfm.json"])
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.layer == "link":
        if len(args.from_root) != len(args.root):
            parser.error("link needs one --from-root per --root")
        for old, new in zip(args.from_root, args.root, strict=True):
            LOGGER.info("%s -> %s: %s", old, new, link_from(old, new, args.files))
        return 0
    client = IngestSources.open().client
    dirs = sorted(d for root in args.root for d in (root / "events").glob("*") if (d / "event.json").is_file())

    def one(d: Path) -> str:
        try:
            return build_dem(d, client, overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001 - one stack's failure is reported; a re-run resumes
            LOGGER.error("%s: %s", d.name, redact(str(exc))[:300])
            return "error"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        statuses = list(pool.map(one, dirs))
    counts: Mapping[str, int] = {s: statuses.count(s) for s in sorted(set(statuses))}
    LOGGER.info("%d stacks: %s", len(dirs), dict(counts))
    return 1 if "error" in counts else 0


if __name__ == "__main__":
    raise SystemExit(main())
