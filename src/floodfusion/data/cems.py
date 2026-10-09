r"""Copernicus EMS Rapid Mapping products as reference labels: the public API, safely.

A CEMS activation (``EMSR664``: the Emilia-Romagna floods of May 2023) is split into
areas of interest (AOIs); each AOI carries products — a first **delineation** (``DEL``,
monitoring 0) and its **monitoring** updates (``DEL`` 1, 2, ...), grading products
(``GRA``) and so on. Each product names the images it was drawn on (sensor, time); a
delineation's ``observedEventA`` layer holds the flood polygons, each tagged with the
source image it came from (``dmg_src_id`` -> the ``source`` table).

Access is public and unauthenticated: a JSON API on ``rapidmapping.emergency.copernicus.eu``
and zip packages behind one redirect to the service's S3 bucket. This module
**allow-lists both hosts**, follows at most that one redirect, caps every download, and
extracts only the GeoJSON layers and the source table from a package — never the
``.lyr``/``.kmz``/``.pdf`` members, and never a member whose name could leave the target
directory.

CEMS delineations are generalised maps from very-high-resolution imagery (often 3 m X-band
SAR) at a stated time: comparing them with a Sentinel-1 map is a comparison between two
*products*, at two instants, with two definitions of "flooded" (minimum mapping unit,
generalisation). That is reported, not hidden: :func:`nearest_product` returns the time
offset with every pairing.

Products: © European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "API_URL",
    "CemsProduct",
    "download_product",
    "fetch_activation",
    "list_products",
    "nearest_product",
    "observed_flood",
    "source_times",
]

LOGGER = logging.getLogger(__name__)

_API_HOST = "rapidmapping.emergency.copernicus.eu"
_S3_HOST = "rapidmapping.s3.amazonaws.com"
API_URL = f"https://{_API_HOST}/backend/dashboard-api/public-activations/"
_MAX_API_BYTES = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_HTTP_TIMEOUT_S = 120
_ACTIVATION = re.compile(r"^EMSR\d{3,4}$")
#: Package members kept: the vector layers as GeoJSON, and the source table.
_KEEP = re.compile(r"^EMSR\d{3,4}_AOI\d{2}_[A-Z]{3}_(PRODUCT|MONIT\d{2})_"
                   r"(observedEventA|areaOfInterestA|imageFootprintA|hydrographyA|hydrographyL)_v\d+\.json$"
                   r"|^EMSR\d{3,4}_AOI\d{2}_[A-Z]{3}_(PRODUCT|MONIT\d{2})_source_v\d+\.dbf$")
#: Some packages wrap their members in ONE folder named after the package; nothing deeper.
_PACKAGE_FOLDER = re.compile(r"^EMSR\d{3,4}_AOI\d{2}_[A-Z]{3}_(PRODUCT|MONIT\d{2})_v\d+$")


# ---------------------------------------------------------------------------
# HTTP: two hosts, one redirect at most, every body capped
# ---------------------------------------------------------------------------
def _check_url(url: str, hosts: Sequence[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise ValueError(f"refusing URL outside {list(hosts)}: {url.split('?', 1)[0][:100]!r}")
    return url


def _read_capped(response: Any, cap: int, url: str) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_content(chunk_size=1 << 20):
        size += len(chunk)
        if size > cap:
            raise OSError(f"{url.split('?', 1)[0]}: larger than {cap} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


#: Transient server answers retried with exponential backoff (the service returns 502s under load).
_RETRY_STATUS = (429, 500, 502, 503, 504)
_RETRIES = 5
_BACKOFF_S = 2.0


class _TransientHTTPError(OSError):
    pass


def _http_get(url: str, *, cap: int, params: Mapping[str, str] | None = None,
              sleep: Callable[[float], None] | None = None) -> bytes:
    """GET with retries on transient errors (429/5xx): ``_RETRIES`` attempts, doubling backoff."""
    import time

    sleep = sleep or time.sleep
    for attempt in range(_RETRIES):
        try:
            return _http_get_once(url, cap=cap, params=params)
        except _TransientHTTPError as exc:
            if attempt == _RETRIES - 1:
                raise
            wait = _BACKOFF_S * 2**attempt
            LOGGER.warning("%s — retrying in %.0f s (%d/%d)", exc, wait, attempt + 1, _RETRIES - 1)
            sleep(wait)
    raise AssertionError("unreachable")


def _http_get_once(url: str, *, cap: int, params: Mapping[str, str] | None = None) -> bytes:
    """GET from the API host; a single redirect is followed only into the CEMS S3 bucket."""
    import requests

    _check_url(url, (_API_HOST,))
    with requests.get(url, params=params, timeout=(10, _HTTP_TIMEOUT_S), allow_redirects=False, stream=True) as r:
        if r.status_code in (301, 302, 303, 307, 308):
            target = _check_url(r.headers.get("Location", ""), (_S3_HOST,))
            with requests.get(target, timeout=(10, _HTTP_TIMEOUT_S), allow_redirects=False, stream=True) as s3:
                if s3.status_code != 200:
                    error = _TransientHTTPError if s3.status_code in _RETRY_STATUS else OSError
                    raise error(f"HTTP {s3.status_code} from the CEMS bucket for {url}")
                return _read_capped(s3, cap, url)
        if r.status_code != 200:
            error = _TransientHTTPError if r.status_code in _RETRY_STATUS else OSError
            raise error(f"HTTP {r.status_code} for {url}")
        return _read_capped(r, cap, url)


Getter = Callable[..., bytes]


# ---------------------------------------------------------------------------
# Activations and products
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CemsProduct:
    activation: str
    aoi: int
    aoi_name: str
    product_type: str            # "DEL" delineation, "GRA" grading, "FEP" first estimate
    monitoring: int              # 0 = first product, k = k-th monitoring update
    download_url: str
    images: tuple[tuple[str, str], ...]   # (sensor, acquisition time ISO-8601), as published

    @property
    def acquisition_times(self) -> tuple[datetime, ...]:
        return tuple(_parse_time(t) for _, t in self.images if t)

    @property
    def tag(self) -> str:
        part = "PRODUCT" if self.monitoring == 0 else f"MONIT{self.monitoring:02d}"
        return f"{self.activation}_AOI{self.aoi:02d}_{self.product_type}_{part}"


def _parse_time(text: str) -> datetime:
    when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def fetch_activation(code: str, *, get: Getter | None = None) -> dict[str, Any]:
    """The public activation record (``EMSR`` + 3-4 digits)."""
    if not _ACTIVATION.match(code):
        raise ValueError(f"not a CEMS rapid-mapping activation code: {code!r}")
    get = get or (lambda url, **kw: _http_get(url, cap=_MAX_API_BYTES, **kw))
    payload = json.loads(get(API_URL, params={"code": code}))
    results = payload.get("results", [payload]) if isinstance(payload, dict) else payload
    if not results or results[0].get("code") != code:
        raise LookupError(f"activation {code} not found")
    return results[0]


def list_products(activation: Mapping[str, Any], *, product_type: str | None = "DEL") -> list[CemsProduct]:
    """Every product (of ``product_type``) of every AOI, sorted by AOI and monitoring number."""
    out = []
    for aoi in activation.get("aois", []):
        for p in aoi.get("products", []):
            if product_type is not None and p.get("type") != product_type:
                continue
            images = tuple((str(i.get("sensorName") or i.get("sensor") or ""), str(i.get("acquisitionTime") or ""))
                           for i in (p.get("images") or []))
            out.append(CemsProduct(
                activation=str(activation["code"]), aoi=int(aoi.get("number", aoi.get("aoiNumber", 0)) or 0),
                aoi_name=str(aoi.get("name", "")), product_type=str(p["type"]),
                monitoring=int(p.get("monitoringNumber") or 0), download_url=str(p.get("downloadPath", "")),
                images=images))
    return sorted(out, key=lambda q: (q.aoi, q.monitoring))


def nearest_product(products: Sequence[CemsProduct], when: datetime) -> tuple[CemsProduct, float]:
    """The product whose (latest) source image is closest in time to ``when``; offset in hours."""
    timed = [(p, max(p.acquisition_times)) for p in products if p.acquisition_times]
    if not timed:
        raise LookupError("no product carries an acquisition time")
    product, t = min(timed, key=lambda pt: abs((pt[1] - when).total_seconds()))
    return product, (when - t).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------
def download_product(product: CemsProduct, dest_root: Path | str, *, get: Getter | None = None) -> Path:
    """Fetch a product package and extract its GeoJSON layers + source table; return the directory.

    Resumable: an existing ``manifest.json`` (written last) means the product is complete.
    """
    out_dir = Path(dest_root) / product.activation / f"AOI{product.aoi:02d}" / product.tag
    if (out_dir / "manifest.json").is_file():
        return out_dir
    get = get or (lambda url, **kw: _http_get(url, cap=_MAX_PACKAGE_BYTES, **kw))
    package = get(_check_url(product.download_url, (_API_HOST,)))
    kept = []
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        out_dir.mkdir(parents=True, exist_ok=True)
        for info in archive.infolist():
            parts = info.filename.split("/")
            name = parts[-1]
            flat = len(parts) == 1 or (len(parts) == 2 and _PACKAGE_FOLDER.match(parts[0]) is not None)
            if info.is_dir() or not flat or not _KEEP.match(name):
                continue  # other paths, directories and unlisted members are never written
            if info.file_size > _MAX_MEMBER_BYTES:
                raise OSError(f"{product.tag}: member {name} is {info.file_size} bytes (cap {_MAX_MEMBER_BYTES})")
            (out_dir / name).write_bytes(archive.read(info))
            kept.append(name)
    if not any("observedEventA" in k for k in kept) and product.product_type == "DEL":
        LOGGER.warning("%s: package has no observedEventA layer (no flood observed?)", product.tag)
    (out_dir / "manifest.json").write_text(json.dumps({
        "product": product.tag, "download_url": product.download_url.split("?", 1)[0], "members": sorted(kept),
        "images": [list(i) for i in product.images], "attribution": "© European Union, Copernicus EMS",
    }, indent=2), encoding="utf-8")
    return out_dir


def _layer(product_dir: Path, layer: str) -> Path | None:
    hits = sorted(product_dir.glob(f"*_{layer}_v*.json"))
    return hits[-1] if hits else None


def source_times(product_dir: Path | str) -> dict[int, datetime]:
    """``{src_id: acquisition time}`` of the post-event images, from the package's source table."""
    import pyogrio

    tables = sorted(Path(product_dir).glob("*_source_v*.dbf"))
    if not tables:
        return {}
    frame = pyogrio.read_dataframe(tables[-1], read_geometry=False)
    out = {}
    for _, row in frame.iterrows():
        day, tm = str(row.get("src_date", "")), str(row.get("source_tm", ""))
        if str(row.get("eventphase", "")).lower() != "post-event" or not re.match(r"^\d{2}/\d{2}/\d{4}$", day):
            continue
        hhmm = re.search(r"T?(\d{2}):(\d{2})", tm)
        d, m, y = (int(v) for v in day.split("/"))
        out[int(row["src_id"])] = datetime(y, m, d, int(hhmm.group(1)) if hhmm else 12,
                                           int(hhmm.group(2)) if hhmm else 0, tzinfo=UTC)
    return out


def observed_flood(product_dir: Path | str) -> Any:
    """The flood polygons (EPSG:4326) with their source image time in ``source_time``."""
    import geopandas as gpd

    product_dir = Path(product_dir)
    path = _layer(product_dir, "observedEventA")
    if path is None:
        return gpd.GeoDataFrame({"source_time": []}, geometry=[], crs="EPSG:4326")
    frame = gpd.read_file(path)
    frame = frame.set_crs(4326) if frame.crs is None else frame.to_crs(4326)  # CEMS GeoJSON is WGS84
    if "event_type" in frame.columns:
        frame = frame[frame["event_type"].astype(str).str.contains("Flood", case=False)]
    times = source_times(product_dir)
    frame["source_time"] = [times.get(int(s)) if s is not None and str(s).isdigit() else None
                            for s in frame.get("dmg_src_id", [None] * len(frame))]
    return frame


def aoi_geometry(product_dir: Path | str) -> Any:
    """The product's area of interest (EPSG:4326 shapely geometry)."""
    import geopandas as gpd

    path = _layer(Path(product_dir), "areaOfInterestA")
    if path is None:
        raise FileNotFoundError(f"{product_dir}: no areaOfInterestA layer")
    frame = gpd.read_file(path)
    frame = frame.set_crs(4326) if frame.crs is None else frame.to_crs(4326)
    return frame.geometry.union_all()
