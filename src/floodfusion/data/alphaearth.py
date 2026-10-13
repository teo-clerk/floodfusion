r"""AlphaEarth Foundations annual embeddings as a land-cover prior (v1.2 context sidecar ``aef.tif``).

**What the embeddings are.** Google and Google DeepMind's annual Satellite Embedding: 64 dimensions per 10 m pixel,
summarising a year of multi-sensor observations. The Source Cooperative mirror (``tge-labs/aef``) is public with no
account, CC BY 4.0. Required attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind."

**Storage format.**
- int8 values, nodata −128, de-quantised as ``(v / 127.5)² · sign(v)``; vectors have unit length.
- One 8192² TIFF per UTM tile, stored bottom-up, with 1024² blocks, band-interleaved, and overviews ×2–×16.
- The ``.vrt`` beside each file points through ``/vsis3`` (S3), so the TIFF is read over HTTPS directly. The
  bottom-up geotransform is a valid affine, and the warp onto our north-up grid handles it. Orientation is checked
  empirically by :func:`orientation_check`.

**Which year.** The year before the event:
``year(post-event acquisition − 60 days) − 1``, clamped to 2025. An event whose prior year is before 2017 gets no
layer, and is marked missing, so the flood can never leak into its own prior.

**What the sidecar holds.**
- The 20 m overview is read: a land-cover prior needs no 10 m detail, and it reads 4× fewer bytes.
- Its bands are read in parallel: 64 sequential range requests per window take over a minute.
- It is warped onto the stack grid (nearest) and projected on a fixed 8-component PCA. The PCA is fitted once on a
  seeded sample of training stacks and committed in ``configs/datasets/aef_pca_v12.json``.
- Stored as int8 ``round(pc · 63.5)``, nodata −128.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.data.chipping import AnalysisGrid

__all__ = ["AEF_SCALE", "FIRST_YEAR", "LAST_YEAR", "N_COMPONENTS", "aef_year", "build_aef", "dequantise",
           "files_for", "fit_pca", "https_url", "main", "orientation_check", "read_embeddings"]

LOGGER = logging.getLogger(__name__)

_HOST = "data.source.coop"
_S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
FIRST_YEAR, LAST_YEAR = 2017, 2025
N_COMPONENTS = 8
#: Stored int8 = round(pc * AEF_SCALE); |pc| <= 2 for centred unit vectors (sidecars.AEF_STORE_SCALE must match).
AEF_SCALE = 63.5
NODATA = -128
_OVERVIEW = 0          # rasterio's first overview: x2, 20 m
_VSI = "/vsicurl/"     # GDAL's HTTP reader (tests read a local file through the same warp)
_THREADS = 16
SIDECAR = "aef.tif"


def https_url(s3_path: str) -> str:
    """The HTTPS URL of an index ``path`` (``s3://us-west-2.opendata.source.coop/tge-labs/aef/...``)."""
    from floodfusion.data.http import check_https_url

    if not str(s3_path).startswith(_S3_PREFIX):
        raise ValueError(f"not an AlphaEarth index path: {str(s3_path)[:80]!r}")
    return check_https_url(f"https://{_HOST}/{s3_path[len(_S3_PREFIX):]}", hosts=(_HOST,),
                           path_prefixes=("/tge-labs/aef/v1/annual/",))


def aef_year(post: datetime) -> int | None:
    """The embedding year of an event whose post-event acquisition is ``post`` (None before 2017's coverage)."""
    year = (post - timedelta(days=60)).year - 1
    if year < FIRST_YEAR:
        return None
    return min(year, LAST_YEAR)


def dequantise(raw: np.ndarray) -> np.ndarray:
    """int8 -> float32 embedding; nodata (-128) -> NaN on every band of that pixel."""
    v = raw.astype(np.float32)
    out = np.sign(v) * (v / 127.5) ** 2
    missing = (raw == NODATA).any(axis=0)
    out[:, missing] = np.nan
    return out


def files_for(index: Any, year: int, lonlat_bounds: Sequence[float]) -> list[str]:
    """Index paths of ``year`` whose WGS84 footprint meets ``lonlat_bounds`` (west, south, east, north)."""
    w, s, e, n = lonlat_bounds
    rows = index[(index["year"] == year) & (index["wgs84_west"] < e) & (index["wgs84_east"] > w)
                 & (index["wgs84_south"] < n) & (index["wgs84_north"] > s)]
    return sorted(rows["path"].tolist())


def _read_band(url: str, band: int, grid: AnalysisGrid) -> np.ndarray:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
           "GDAL_HTTP_TIMEOUT": 120, "GDAL_HTTP_MAX_RETRY": 4, "GDAL_HTTP_RETRY_DELAY": 3}
    with rasterio.Env(**env), rasterio.open(_VSI + url, overview_level=_OVERVIEW) as src, \
            WarpedVRT(src, crs=CRS.from_epsg(grid.epsg), transform=Affine(*grid.transform), width=grid.width,
                      height=grid.height, resampling=Resampling.nearest, src_nodata=NODATA, nodata=NODATA) as vrt:
        return vrt.read(band)


def read_embeddings(paths: Sequence[str], grid: AnalysisGrid) -> np.ndarray:
    """``(64, H, W)`` float32 embeddings on ``grid`` (NaN where no file has a value), files mosaicked first-valid."""
    from concurrent.futures import ThreadPoolExecutor

    out = np.full((64, grid.height, grid.width), NODATA, dtype=np.int8)
    for path in paths:
        url = https_url(path)
        with ThreadPoolExecutor(max_workers=_THREADS) as pool:
            bands = list(pool.map(lambda b, u=url: _read_band(u, b, grid), range(1, 65)))
        layer = np.stack(bands).astype(np.int8)
        gap = (out == NODATA).any(axis=0) & ~(layer == NODATA).any(axis=0)
        out[:, gap] = layer[:, gap]
    return dequantise(out)


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------
def _stack_grid(event_dir: Path) -> tuple[AnalysisGrid, dict[str, Any]]:
    meta = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    transform = meta["transform"] if isinstance(meta["transform"], list) else json.loads(meta["transform"])
    return AnalysisGrid(epsg=int(meta["epsg"]), transform=tuple(float(v) for v in transform),
                        width=int(meta["width"]), height=int(meta["height"])), meta


def _post(meta: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(meta["s1_post_datetime"]).replace("Z", "+00:00"))


def _lonlat(grid: AnalysisGrid) -> tuple[float, float, float, float]:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    return tuple(transform_bounds(CRS.from_epsg(grid.epsg), CRS.from_epsg(4326), *grid.bounds, densify_pts=21))


def stack_embeddings(event_dir: Path, index: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The stack's embeddings (or None) and the provenance record (year, files)."""
    grid, meta = _stack_grid(event_dir)
    year = aef_year(_post(meta))
    if year is None:
        return None, {"status": "no AlphaEarth year before the event", "year": None, "files": []}
    paths = files_for(index, year, _lonlat(grid))
    if not paths:
        return None, {"status": f"no AlphaEarth file of {year}", "year": year, "files": []}
    return read_embeddings(paths, grid), {"status": "ok", "year": year, "files": paths}


def fit_pca(samples: np.ndarray, n: int = N_COMPONENTS) -> dict[str, Any]:
    """Mean and the top ``n`` principal axes of ``samples`` (pixels x 64), as JSON-ready lists."""
    x = np.asarray(samples, dtype=np.float64)
    mean = x.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(values)[::-1][:n]
    comps = vectors[:, order].T
    comps *= np.sign(comps[np.arange(n), np.abs(comps).argmax(axis=1)])[:, None]   # deterministic sign
    return {"mean": mean.tolist(), "components": comps.tolist(), "explained_variance": values[order].tolist(),
            "explained_ratio": (values[order] / values.sum()).tolist(), "n_samples": int(len(x))}


def project(emb: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    """``(n, H, W)`` int8 sidecar values of ``(64, H, W)`` embeddings; NaN pixels -> nodata."""
    mean, comps = np.asarray(pca["mean"]), np.asarray(pca["components"])
    flat = emb.reshape(64, -1).T
    pcs = (flat - mean) @ comps.T
    out = np.clip(np.round(pcs * AEF_SCALE), -127, 127)
    out[~np.isfinite(pcs).all(axis=1)] = NODATA
    return out.T.reshape(len(comps), *emb.shape[1:]).astype(np.int8)


def build_aef(event_dir: Path, index: Any, pca: dict[str, Any], *, overwrite: bool = False) -> str:
    """Write ``aef.tif`` (int8 PCs, nodata -128) and ``aef.json`` beside one stack."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from floodfusion.data.records import atomic_write_bytes

    out, record_path = event_dir / SIDECAR, event_dir / "aef.json"
    if out.is_file() and record_path.is_file() and not overwrite:
        return "done-before"
    emb, record = stack_embeddings(event_dir, index)
    if emb is None:
        out.unlink(missing_ok=True)
        atomic_write_bytes(record_path, json.dumps(record).encode())
        return str(record["status"])
    grid, _ = _stack_grid(event_dir)
    pcs = project(emb, pca)
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", dtype="int8", count=pcs.shape[0], height=grid.height,
                       width=grid.width, crs=CRS.from_epsg(grid.epsg), transform=Affine(*grid.transform),
                       nodata=NODATA, compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(pcs)
    tmp.replace(out)
    record.update(valid_fraction=float(np.isfinite(emb[0]).mean()), pca_sha256=pca_digest(pca))
    atomic_write_bytes(record_path, json.dumps(record).encode())
    return "ok"


def pca_digest(pca: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: pca[k] for k in ("mean", "components")}).encode()).hexdigest()


def orientation_check(emb: np.ndarray, water: np.ndarray) -> dict[str, float]:
    """How well a nearest-centroid rule separates ``water`` from land in ``emb`` as read, and flipped upside down.

    AlphaEarth separates open water from land easily. Read the right way up the score is high; flipped, the
    embeddings no longer sit over their pixels and it collapses towards 0.5. The centroids come from one half of a
    checkerboard and are scored on the other, so 64 noisy dimensions cannot fit chance structure.
    """
    rows, cols = np.indices(water.shape)
    fit = (rows + cols) % 2 == 0          # centroids from one half of a checkerboard, scored on the other

    def auc(e: np.ndarray) -> float:
        ok = np.isfinite(e).all(axis=0)
        w, land = water & ok, ~water & ok
        if (w & fit).sum() < 25 or (land & fit).sum() < 25 or (w & ~fit).sum() < 25 or (land & ~fit).sum() < 25:
            return float("nan")
        cw, cl = e[:, w & fit].mean(axis=1), e[:, land & fit].mean(axis=1)
        score = np.tensordot(cw - cl, e, axes=1)
        s_w, s_l = score[w & ~fit], score[land & ~fit]
        ranks = np.argsort(np.argsort(np.concatenate([s_w, s_l]))) + 1.0
        return float((ranks[: len(s_w)].sum() - len(s_w) * (len(s_w) + 1) / 2) / (len(s_w) * len(s_l)))

    return {"auc_as_read": auc(emb), "auc_flipped": auc(emb[:, ::-1, :])}


def _water(event_dir: Path) -> np.ndarray:
    import rasterio

    from floodfusion.config import ValidityFlag

    with rasterio.open(event_dir / "stack.tif") as src:
        flags = src.read(14)
    return (flags.astype(np.int64) & int(ValidityFlag.PERMANENT_WATER)) != 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaEarth context sidecars (aef.tif) beside event stacks.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "fit", "build"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, nargs="+", required=True)
        p.add_argument("--index", type=Path, default=Path("data/aef_raw/aef_index.parquet"))
        p.add_argument("--pca", type=Path, default=Path("configs/datasets/aef_pca_v12.json"))
    sub.choices["check"].add_argument("--n", type=int, default=8)
    sub.choices["fit"].add_argument("--n-stacks", type=int, default=40)
    sub.choices["fit"].add_argument("--per-stack", type=int, default=1000)
    sub.choices["build"].add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import pandas as pd

    index = pd.read_parquet(args.index, columns=["path", "year", "wgs84_west", "wgs84_south", "wgs84_east",
                                                 "wgs84_north"])
    dirs = sorted(d for root in args.root for d in (root / "events").glob("*") if (d / "event.json").is_file())
    rng = np.random.default_rng(int(hashlib.sha256(b"aef-v12").hexdigest()[:8], 16))
    if args.cmd == "check":
        candidates = [d for d in dirs if _water(d).mean() > 0.02]
        for d in [candidates[i] for i in sorted(rng.permutation(len(candidates))[: args.n])]:
            emb, rec = stack_embeddings(d, index)
            if emb is not None:
                LOGGER.info("%s %s: %s", d.name, rec["year"], orientation_check(emb, _water(d)))
        return 0
    if args.cmd == "fit":
        picked = [dirs[i] for i in sorted(rng.permutation(len(dirs)))]
        samples, used = [], []
        for d in picked:
            if len(used) == args.n_stacks:
                break
            emb, rec = stack_embeddings(d, index)
            if emb is None:
                continue
            flat = emb.reshape(64, -1).T
            flat = flat[np.isfinite(flat).all(axis=1)]
            if len(flat) < args.per_stack:
                continue
            samples.append(flat[rng.choice(len(flat), args.per_stack, replace=False)])
            used.append(d.name)
            LOGGER.info("sampled %s (%d/%d)", d.name, len(used), args.n_stacks)
        pca = {**fit_pca(np.concatenate(samples)), "stacks": used, "scale": AEF_SCALE}
        args.pca.write_text(json.dumps(pca, indent=1) + "\n", encoding="utf-8")
        LOGGER.info("PCA from %d stacks: explained %s", len(used), np.round(pca["explained_ratio"], 3).tolist())
        return 0
    from concurrent.futures import ThreadPoolExecutor

    pca = json.loads(args.pca.read_text(encoding="utf-8"))

    def one(d: Path) -> str:
        try:
            return build_aef(d, index, pca)
        except Exception as exc:  # noqa: BLE001 - reported; a re-run resumes
            from floodfusion.data.http import redact

            LOGGER.error("%s: %s", d.name, redact(str(exc))[:300])
            return "error"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        statuses = list(pool.map(one, dirs))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    LOGGER.info("%d stacks: %s", len(dirs), counts)
    return 1 if "error" in counts else 0


if __name__ == "__main__":
    raise SystemExit(main())
