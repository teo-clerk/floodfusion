r"""Readers that bring each public source onto the analysis grid — resampled **once**.

All open, all credential-free through Microsoft Planetary Computer (anonymous SAS signing):

=================== ============================ ==================== =====================
layer               collection / asset           kernel               notes
=================== ============================ ==================== =====================
gamma0 VV, VH       ``sentinel-1-rtc`` vv, vh    bilinear, LINEAR     Route B: RTC already
                                                 power                applied (docs/04 9)
S1 geometry         ``sentinel-1-grd``           —                    heading, incidence
                    ``schema-product-*``                              tie points
B3, B8, B11         ``sentinel-2-l2a`` B03 B08   bilinear             offset from
                    B11                                               s2:processing_baseline
SCL                 ``sentinel-2-l2a`` SCL       NEAREST (classes)    quality mask only
DEM                 ``cop-dem-glo-30`` data      bilinear             EGM2008 heights
permanent water     ``jrc-gsw`` occurrence       bilinear, then >=80%
land cover          ``esa-worldcover`` map       NEAREST (classes)    R4 caveats, analysis
=================== ============================ ==================== =====================

Every reader takes plain hrefs, so local GeoTIFFs work identically (the offline tests do
exactly that); remote Planetary Computer blob hrefs are signed just before reading.

The RTC checklist of ``docs/04`` section 9 is applied, not assumed:
:func:`verify_rtc_linear_power` refuses a layer that looks like dB when linear power is
expected — "a collection in linear power read as dB produces values around 1e-2 that are
plausible to a careless normaliser and catastrophic to a threshold".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from floodfusion.data.chipping import AnalysisGrid, reproject_to_grid

__all__ = [
    "ALLOWED_REMOTE_HOSTS",
    "ALLOWED_REMOTE_SUFFIXES",
    "asset_href",
    "check_href",
    "read_dem",
    "read_jrc_occurrence",
    "read_s1_rtc",
    "read_s2_reflectance",
    "read_s2_scl",
    "read_worldcover",
    "s2_offsets",
    "signed",
    "verify_rtc_linear_power",
]

LOGGER = logging.getLogger(__name__)


#: Remote hosts an asset href may point at: Planetary Computer's blob storage and API.
ALLOWED_REMOTE_SUFFIXES = (".blob.core.windows.net",)
ALLOWED_REMOTE_HOSTS = ("planetarycomputer.microsoft.com",)


def check_href(href: str) -> str:
    """Refuse asset hrefs a STAC item should never carry; return the href unchanged.

    Remote hrefs must be ``https`` to Planetary Computer storage (an ``http`` annotation could
    be swapped in transit and silently rotate every shadow mask). GDAL virtual paths
    (``/vsi*``) and inline VRT XML are refused: they let an item make GDAL open arbitrary
    URLs or local files. Local paths are allowed — offline fixtures and local mirrors.
    """
    from urllib.parse import urlparse

    text = str(href).strip()
    if text.startswith("/vsi") or text.lstrip().startswith("<"):
        raise ValueError(f"refusing GDAL virtual path or inline XML as an asset href: {text[:60]!r}")
    parsed = urlparse(text)
    if parsed.scheme in ("", "file"):
        return text
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host in ALLOWED_REMOTE_HOSTS or host.endswith(ALLOWED_REMOTE_SUFFIXES)):
        raise ValueError(f"refusing asset href outside Planetary Computer over https: {parsed.scheme}://{host}")
    return text


def signed(href: str) -> str:
    """Check, then sign a Planetary Computer blob href (anonymous SAS token)."""
    href = check_href(href)
    if href.startswith("https://") and "?" not in href:
        import planetary_computer

        return planetary_computer.sign(href)
    return href.removeprefix("file://")


def _item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    return item.to_dict(include_self_link=False, transform_hrefs=False)


def asset_href(item: Any, *names: str) -> str:
    """The href of the first asset of ``item`` whose key is in ``names`` (case-insensitive)."""
    assets = _item_dict(item).get("assets", {})
    lower = {k.lower(): v for k, v in assets.items()}
    for name in names:
        if name.lower() in lower:
            return str(lower[name.lower()]["href"])
    raise KeyError(f"item {_item_dict(item).get('id')} has none of the assets {names}; has {sorted(assets)}")


def _mosaic(hrefs: Sequence[str], grid: AnalysisGrid, *, kind: str, src_nodata: float | None = None) -> np.ndarray:
    """First-valid mosaic of several sources onto the grid (frames of one acquisition, tiles)."""
    out: np.ndarray | None = None
    for href in hrefs:
        layer = reproject_to_grid(signed(href), grid, kind=kind, src_nodata=src_nodata, dtype="float32")
        out = layer if out is None else np.where(np.isfinite(out), out, layer)
    if out is None:
        raise ValueError("no source hrefs to mosaic")
    return out.astype(np.float64)


def verify_rtc_linear_power(values: np.ndarray, *, name: str = "gamma0") -> dict[str, float]:
    """``docs/04`` section 9: confirm a layer is linear power, and report its QA numbers.

    Linear gamma0 over land and water sits roughly in [1e-4, 1]; dB sits in [-35, +10]
    with a negative median. Raises ``ValueError`` if the layer looks like dB.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError(f"{name}: no finite pixels")
    median = float(np.median(finite))
    if median < 0 or median > 10:
        raise ValueError(f"{name}: median {median:.3g} looks like dB, but linear power is expected")
    db = 10 * np.log10(np.clip(finite, 1e-6, None))
    counts, edges = np.histogram(db, bins=80, range=(-40, 10))
    return {
        "finite_fraction": float(finite.size / v.size),
        "median_linear": median,
        "median_db": float(np.median(db)),
        "histogram_mode_db": float(edges[np.argmax(counts)] + 0.5 * (edges[1] - edges[0])),
        "non_positive_fraction": float(np.mean(finite <= 0)),
    }


def read_s1_rtc(items: Sequence[Any], grid: AnalysisGrid, polarisation: str) -> tuple[np.ndarray, dict[str, float]]:
    """gamma0 in **dB** on the grid, from the RTC frames of one acquisition.

    The bilinear resampling happens on **linear power** (averaging in dB is biased low,
    ``docs/04`` section 3.4); dB conversion is the last step. Non-positive and nodata
    pixels become NaN.
    """
    hrefs = [asset_href(it, polarisation.lower()) for it in items]
    linear = _mosaic(hrefs, grid, kind="continuous")  # the file's own nodata (-32768 on PC)
    linear[~(linear > 0)] = np.nan
    qa = verify_rtc_linear_power(linear, name=f"gamma0_{polarisation}")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(linear), qa


def s2_offsets(item: Any) -> tuple[float, float]:
    """(BOA_ADD_OFFSET, QUANTIFICATION_VALUE) for an L2A item, from its processing baseline.

    From processing baseline 04.00 (January 2022) L2A DN carry an offset of -1000;
    earlier products have none (``docs/05`` section 7.2).
    """
    props = _item_dict(item).get("properties", {})
    if "s2:boa_add_offset" in props:
        return float(props["s2:boa_add_offset"]), float(props.get("s2:boa_quantification_value", 10000.0))
    baseline = str(props.get("s2:processing_baseline", "00.00"))
    try:
        major = float(baseline)
    except ValueError:
        major = 0.0
    return (-1000.0 if major >= 4.0 else 0.0), 10000.0


def read_s2_scl(items: Sequence[Any], grid: AnalysisGrid) -> np.ndarray:
    """The SCL of one overpass (all its tiles) on the grid — **nearest-neighbour, always**.

    Bilinear interpolation of class 3 (cloud shadow) and 9 (cloud high probability) yields
    6 = WATER (``docs/02`` section 8). 0 = no data.
    """
    scl = _mosaic([asset_href(it, "SCL") for it in items], grid, kind="categorical", src_nodata=0)
    return np.nan_to_num(scl, nan=0.0).astype(np.uint8)


def read_s2_reflectance(items: Sequence[Any], grid: AnalysisGrid) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Surface reflectance ``(3, H, W)`` (B3, B8, B11) and the SCL ``(H, W)`` of one overpass.

    Each tile's DN are converted with **its own** offset (``s2_offsets``) before the
    first-valid mosaic, so tiles of different processing baselines cannot be mixed in DN.
    B11 (20 m) is upsampled bilinearly; DN 0 is nodata.
    """
    if not items:
        raise ValueError("no Sentinel-2 items to read")
    bands = np.full((3, grid.height, grid.width), np.nan)
    offsets = {}
    for item in items:
        offset, quantification = s2_offsets(item)
        offsets[_item_dict(item).get("id", "?")] = offset
        for b, name in enumerate(("B03", "B08", "B11")):
            dn = reproject_to_grid(signed(asset_href(item, name)), grid, kind="continuous", src_nodata=0,
                                   dtype="float32").astype(np.float64)
            rho = np.where(np.isfinite(dn) & (dn > 0), (dn + offset) / quantification, np.nan)
            bands[b] = np.where(np.isfinite(bands[b]), bands[b], rho)
    return bands, read_s2_scl(items, grid), {"boa_add_offset_by_item": offsets}


def read_dem(items: Sequence[Any], grid: AnalysisGrid) -> np.ndarray:
    """Copernicus GLO-30 heights (EGM2008) mosaicked onto the grid, bilinear.

    Reprojecting to the metric grid **first** is what makes the Horn slope meaningful
    (``docs/03`` section 4.2); the transform is read from each tile, so the non-square
    geographic pixels above 50 N are handled by the warp, never assumed.
    """
    return _mosaic([asset_href(it, "data", "dem") for it in items], grid, kind="continuous")


def read_jrc_occurrence(items: Sequence[Any], grid: AnalysisGrid) -> np.ndarray:
    """JRC Global Surface Water occurrence (percent); 255 = nodata -> NaN."""
    occ = _mosaic([asset_href(it, "occurrence") for it in items], grid, kind="continuous", src_nodata=255)
    occ[occ > 100] = np.nan
    return occ


def read_worldcover(items: Sequence[Any], grid: AnalysisGrid) -> np.ndarray:
    """ESA WorldCover classes (uint8), nearest-neighbour; 0 = unknown."""
    lc = _mosaic([asset_href(it, "map") for it in items], grid, kind="categorical", src_nodata=0)
    return np.nan_to_num(lc, nan=0.0).astype(np.uint8)
