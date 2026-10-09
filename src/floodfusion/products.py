r"""The product as an operational package: GFM-style layers as COGs, described by a STAC Item.

The layers follow the structure of the Copernicus EMS **Global Flood Monitoring** (GFM)
product — the Sentinel-1 flood service this map would be compared with — so a user of one
can read the other:

======================== ======= ============================================================
layer                    dtype   content
======================== ======= ============================================================
``observed_flood``       uint8   1 flood at τ*, 0 not flood, 255 no information
``observed_water``       uint8   1 flood **or** permanent water, 0 dry, 255 no information
``reference_water``      uint8   1 permanent water (JRC GSW occurrence >= 80 %), 0 otherwise
``exclusion_mask``       uint8   1 where SAR flood mapping is not feasible: no data, radar
                                 shadow, layover, dense urban, dense forest (GFM's exclusion
                                 classes); 0 elsewhere
``likelihood``           uint8   calibrated flood probability x 100 (0-100), 255 no data
``advisory_flags``       uint8   bit 0 LOW_CONFIDENCE (decision flips if the cost ratio
                                 changes 1.5x), bit 1 STEEP_SLOPE, bit 2 OPTICAL_USED
``flags``                uint16  the full bit field of :data:`floodfusion.predict.PRODUCT_FLAGS`
======================== ======= ============================================================

Excluded pixels keep their likelihood: the exclusion mask says *do not trust this*, it
does not erase the evidence. Every file is a Cloud-Optimized GeoTIFF (tiled, deflate,
overviews — nearest for classes, average for the likelihood), readable by a web map
without downloading it. ``item.json`` is a STAC 1.0 Item: acquisition time and footprint,
the ``projection``, ``raster``, ``sat``, ``sar`` and ``processing`` extensions, links
``derived_from`` every input scene, and the git SHA, configuration and model hashes that
reproduce it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.predict import PRODUCT_FLAGS, product_classes

__all__ = ["GFM_LAYERS", "STAC_EXTENSIONS", "gfm_layers", "stac_item", "write_cog", "write_package"]

LOGGER = logging.getLogger(__name__)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_EXTENSIONS = (
    "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
    "https://stac-extensions.github.io/sat/v1.0.0/schema.json",
    "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
    "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
)
_NO_INFO = ("NODATA", "LAYOVER", "SHADOW", "OUTSIDE_AOI")
_EXCLUDED = ("NODATA", "LAYOVER", "SHADOW", "DENSE_URBAN", "DENSE_FOREST")
_ADVISORY = {"LOW_CONFIDENCE": 0, "STEEP_SLOPE": 1, "OPTICAL_USED": 2}
NO_DATA_U8 = 255


@dataclass(frozen=True)
class Layer:
    name: str
    data: np.ndarray
    dtype: str
    nodata: int | None
    resampling: str
    description: str


GFM_LAYERS = ("observed_flood", "observed_water", "reference_water", "exclusion_mask", "likelihood",
              "advisory_flags", "flags")


def _bits(flags: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return (flags & sum(1 << PRODUCT_FLAGS[n] for n in names)) > 0


def gfm_layers(prob: np.ndarray, flags: np.ndarray, *, tau: float) -> dict[str, Layer]:
    """The seven layers from the calibrated probability and the product flag bit field."""
    classes = product_classes(prob, flags, tau)
    no_info = classes == 255
    flood = np.where(no_info, NO_DATA_U8, classes == 1).astype(np.uint8)
    water = np.where(no_info, NO_DATA_U8, (classes == 1) | (classes == 2)).astype(np.uint8)
    reference = _bits(flags, ("PERMANENT_WATER",)).astype(np.uint8)
    exclusion = _bits(flags, _EXCLUDED).astype(np.uint8)
    likelihood = np.where(np.isfinite(prob), np.clip(np.rint(100.0 * np.nan_to_num(prob)), 0, 100),
                          NO_DATA_U8).astype(np.uint8)
    advisory = np.zeros(flags.shape, dtype=np.uint8)
    for name, bit in _ADVISORY.items():
        advisory |= (_bits(flags, (name,)).astype(np.uint8) << bit)
    return {
        "observed_flood": Layer("observed_flood", flood, "uint8", NO_DATA_U8, "nearest",
                                f"1 flood at tau*={tau:.3f}, 0 not flood, 255 no information"),
        "observed_water": Layer("observed_water", water, "uint8", NO_DATA_U8, "nearest",
                                "1 flood or permanent water, 0 dry, 255 no information"),
        "reference_water": Layer("reference_water", reference, "uint8", None, "nearest",
                                 "1 permanent water (JRC GSW occurrence >= 80 %)"),
        "exclusion_mask": Layer("exclusion_mask", exclusion, "uint8", None, "nearest",
                                "1 SAR flood mapping not feasible: " + ", ".join(n.lower() for n in _EXCLUDED)),
        "likelihood": Layer("likelihood", likelihood, "uint8", NO_DATA_U8, "average",
                            "calibrated flood probability x 100; 255 no data"),
        "advisory_flags": Layer("advisory_flags", advisory, "uint8", None, "nearest",
                                "bit 0 low confidence, bit 1 steep slope, bit 2 optical used"),
        "flags": Layer("flags", flags.astype(np.uint16), "uint16", None, "nearest",
                       "bit field " + json.dumps(PRODUCT_FLAGS)),
    }


def write_cog(path: Path, layer: Layer, profile: Mapping[str, Any], tags: Mapping[str, Any]) -> Path:
    """One band as a Cloud-Optimized GeoTIFF (written to memory, then copied by the COG driver)."""
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy as rio_copy

    base = {k: profile[k] for k in ("width", "height", "crs", "transform")}
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", count=1, dtype=layer.dtype, nodata=layer.nodata, **base) as ds:
            ds.write(layer.data.astype(layer.dtype), 1)
            ds.set_band_description(1, layer.description)
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
        with mem.open() as ds:
            tmp = path.with_name(f".{path.name}.tmp")
            rio_copy(ds, tmp, driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=256,
                     OVERVIEW_RESAMPLING=layer.resampling.upper(), BIGTIFF="IF_SAFER")
    tmp.replace(path)
    return path


def _pixel_size(profile: Mapping[str, Any]) -> float:
    a, _, _, _, e, _ = tuple(profile["transform"])[:6]
    if abs(abs(a) - abs(e)) > 1e-9 * abs(a):
        raise ValueError(f"non-square pixels {a} x {e}: the product grid is square by construction")
    return float(abs(a))


def _footprint(profile: Mapping[str, Any]) -> tuple[dict[str, Any], list[float]]:
    from rasterio.warp import transform_bounds

    a, _, c, _, e, f = tuple(profile["transform"])[:6]
    xmin, ymax = c, f
    xmax, ymin = c + a * profile["width"], f + e * profile["height"]
    w, s, e_, n = transform_bounds(profile["crs"], "EPSG:4326", xmin, ymin, xmax, ymax)
    ring = [[w, s], [e_, s], [e_, n], [w, n], [w, s]]
    return {"type": "Polygon", "coordinates": [ring]}, [w, s, e_, n]


def stac_item(item_id: str, profile: Mapping[str, Any], assets: Mapping[str, Path], layers: Mapping[str, Layer],
              meta: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    """A STAC 1.0 Item (as a dict) describing the package; asset hrefs are relative."""
    from rasterio.crs import CRS

    geometry, bbox = _footprint(profile)
    when = str(meta.get("s1_post_datetime") or datetime.now(UTC).isoformat())
    epsg = CRS.from_user_input(profile["crs"]).to_epsg()
    inputs = ([("sentinel-1-rtc", i) for i in meta.get("s1_post_frames", [])]
              + [("sentinel-1-rtc", i) for i in meta.get("s1_pre_frames", [])]
              + [("sentinel-2-l2a", i) for i in meta.get("s2_items", [])])
    properties = {
            "datetime": when, "title": f"floodfusion flood extent, {meta.get('event_id', '')}",
            "description": "Observed flood extent from Sentinel-1 (and Sentinel-2 where clear), GFM-style layers.",
            "proj:code": f"EPSG:{epsg}", "proj:shape": [int(profile["height"]), int(profile["width"])],
            "proj:transform": list(tuple(profile["transform"])[:6]),
            "sat:relative_orbit": meta.get("relative_orbit"), "sat:orbit_state": meta.get("orbit_state"),
            "sar:instrument_mode": "IW", "sar:frequency_band": "C", "sar:polarizations": ["VV", "VH"],
            "sar:product_type": "RTC",
            "processing:level": "L3", "processing:software": {"floodfusion": str(provenance.get("version", ""))},
            "processing:lineage": "Planetary Computer sentinel-1-rtc pre/post pair on one relative orbit; "
                                  "calibrated dual-branch SAR-optical segmentation; tau* from a cost ratio",
            "floodfusion:tau_star": provenance.get("tau_star"), "floodfusion:git_sha": provenance.get("git_sha"),
            "floodfusion:config_sha256": provenance.get("config_sha256"),
            "floodfusion:model_sha256": provenance.get("model_sha256"),
            "floodfusion:calibration": provenance.get("calibration"),
            "floodfusion:sar_only": bool(provenance.get("sar_only", False)),
    }
    # Extension fields are typed by their schemas: an unknown value is omitted, never null.
    properties = {k: v for k, v in properties.items() if v is not None}

    def band(layer: Layer) -> dict[str, Any]:
        out: dict[str, Any] = {"data_type": layer.dtype, "spatial_resolution": _pixel_size(profile)}
        if layer.nodata is not None:
            out["nodata"] = layer.nodata
        return out

    # Declare an extension only when one of its fields is present (its schema requires one).
    extensions = [u for u in STAC_EXTENSIONS if "/sat/" not in u or any(k.startswith("sat:") for k in properties)]
    return {
        "type": "Feature", "stac_version": "1.0.0", "stac_extensions": extensions, "id": item_id,
        "geometry": geometry, "bbox": bbox, "properties": properties,
        "links": [{"rel": "derived_from", "href": f"{PC_STAC}/collections/{c}/items/{i}", "type": "application/geo+json"}
                  for c, i in inputs],
        "assets": {
            name: {"href": f"./{path.name}", "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                   "roles": ["data"], "title": name.replace("_", " "), "description": layers[name].description,
                   "raster:bands": [band(layers[name])]}
            for name, path in assets.items()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_package(out_dir: Path | str, prob: np.ndarray, flags: np.ndarray, *, tau: float, profile: Mapping[str, Any],
                  meta: Mapping[str, Any], provenance: Mapping[str, Any], weights: Path | None = None) -> Path:
    """Write the seven COG layers and ``item.json`` into ``out_dir``; return the Item path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {**provenance, "tau_star": tau}
    if weights is not None and weights.is_file():
        provenance["model_sha256"] = _sha256(weights)
    layers = gfm_layers(prob, flags, tau=tau)
    tags = {"tau_star": tau, "git_sha": provenance.get("git_sha"), "attribution":
            "Contains modified Copernicus Sentinel data processed by Planetary Computer"}
    assets = {name: write_cog(out_dir / f"{name}.tif", layer, profile, tags) for name, layer in layers.items()}
    item_id = f"{meta.get('event_id', 'event')}_{str(meta.get('s1_post_datetime', ''))[:19].replace(':', '')}"
    item = stac_item(item_id, profile, assets, layers, meta, provenance)
    path = out_dir / "item.json"
    path.write_text(json.dumps(item, indent=2, default=str), encoding="utf-8")
    flood = layers["observed_flood"].data
    px_km2 = _pixel_size(profile) ** 2 / 1e6
    LOGGER.info("package %s: %.2f km2 flood, %.1f %% excluded", out_dir, float((flood == 1).sum()) * px_km2,
                100.0 * float(layers["exclusion_mask"].data.mean()))
    return path
