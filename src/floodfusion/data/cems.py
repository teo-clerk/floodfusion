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

__all__ = [
    "API_URL",
    "CemsProduct",
    "download_product",
    "fetch_activation",
    "layer_path",
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
#: The whole extraction is bounded too: a small package can declare many large members.
_MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
_MAX_MEMBERS = 64
_PRODUCT_TYPE = re.compile(r"[A-Z]{3}", re.ASCII)
_HTTP_TIMEOUT_S = 120
_ACTIVATION = re.compile(r"EMSR[0-9]{3,4}", re.ASCII)
#: Package members kept: the vector layers as GeoJSON, and the source table.
_KEEP = re.compile(r"EMSR[0-9]{3,4}_AOI[0-9]{2}_[A-Z]{3}_(PRODUCT|MONIT[0-9]{2})_"
                   r"(?:(observedEventA|areaOfInterestA|imageFootprintA|hydrographyA|hydrographyL)_v[0-9]+\.json"
                   r"|source_v[0-9]+\.dbf)", re.ASCII)
#: Some packages wrap their members in ONE folder named after the package; nothing deeper.
_PACKAGE_FOLDER = re.compile(r"EMSR[0-9]{3,4}_AOI[0-9]{2}_[A-Z]{3}_(PRODUCT|MONIT[0-9]{2})_v[0-9]+", re.ASCII)


# ---------------------------------------------------------------------------
# HTTP: two hosts, one redirect at most, every body capped
# ---------------------------------------------------------------------------
def _check_url(url: str, hosts: Sequence[str]) -> str:
    from floodfusion.data.http import check_https_url

    return check_https_url(url, hosts=hosts)


#: Transient failures (429/5xx, resets, timeouts) are retried with a doubling backoff.
_RETRIES = 5
_BACKOFF_S = 2.0


def _http_get(url: str, *, cap: int, params: Mapping[str, str] | None = None,
              sleep: Callable[[float], None] | None = None) -> bytes:
    """GET from the API host; a single redirect is followed only into the CEMS S3 bucket."""
    import time

    from floodfusion.data.http import capped_get

    return capped_get(url, check=lambda u: _check_url(u, (_API_HOST,)), cap=cap, params=params,
                      timeout_s=_HTTP_TIMEOUT_S, redirect_check=lambda u: _check_url(u, (_S3_HOST,)),
                      retries=_RETRIES, backoff_s=_BACKOFF_S, sleep=sleep or time.sleep)


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
    if not _ACTIVATION.fullmatch(code):
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
            if not _PRODUCT_TYPE.fullmatch(str(p.get("type", ""))):  # it becomes part of a directory name
                raise ValueError(f"unexpected product type {str(p.get('type'))[:20]!r} in the activation record")
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
    kept: list[str] = []
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        out_dir.mkdir(parents=True, exist_ok=True)
        members = [i for i in archive.infolist() if _keep(i)]
        total = sum(i.file_size for i in members)
        if len(members) > _MAX_MEMBERS or total > _MAX_EXTRACTED_BYTES:
            raise OSError(f"{product.tag}: {len(members)} members, {total} bytes declared — over the caps "
                          f"({_MAX_MEMBERS} members, {_MAX_EXTRACTED_BYTES} bytes)")
        for info in members:
            name = info.filename.split("/")[-1]
            if info.file_size > _MAX_MEMBER_BYTES:
                raise OSError(f"{product.tag}: member {name} is {info.file_size} bytes (cap {_MAX_MEMBER_BYTES})")
            _extract(archive, info, out_dir / name)
            kept.append(name)
    if not any("observedEventA" in k for k in kept) and product.product_type == "DEL":
        LOGGER.warning("%s: package has no observedEventA layer (no flood observed?)", product.tag)
    (out_dir / "manifest.json").write_text(json.dumps({
        "product": product.tag, "download_url": product.download_url.split("?", 1)[0], "members": sorted(kept),
        "images": [list(i) for i in product.images], "attribution": "© European Union, Copernicus EMS",
    }, indent=2), encoding="utf-8")
    return out_dir


def _keep(info: zipfile.ZipInfo) -> bool:
    """A listed layer at the package root, or inside ONE package-named folder; nothing else."""
    parts = info.filename.split("/")
    flat = len(parts) == 1 or (len(parts) == 2 and _PACKAGE_FOLDER.fullmatch(parts[0]) is not None)
    return not info.is_dir() and flat and _KEEP.fullmatch(parts[-1]) is not None


def content_matches(name: str, head: bytes) -> bool:
    """Whether a member's first bytes are what its suffix promises: GeoJSON starts with ``{``, a
    dBase table with a known version byte. Anything else (an ``<OGRVRTDataSource>`` above all) is refused."""
    if name.endswith(".json"):
        return head.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] == b"{"
    if name.endswith(".dbf"):
        return bool(head) and head[0] in (0x02, 0x03, 0x04, 0x05, 0x30, 0x31, 0x32, 0x43, 0x63, 0x83, 0x8B, 0x8E, 0xCB,
                                          0xF5, 0xFB)
    return False


def _extract(archive: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> None:
    """Stream one member to ``dest``, stopping at its declared size (a lying header cannot inflate it)."""
    written = 0
    tmp = dest.with_name(f".{dest.name}.tmp")
    with archive.open(info) as src, tmp.open("wb") as out:
        while chunk := src.read(1 << 20):
            if not written and not content_matches(dest.name, chunk[:64]):
                raise OSError(f"{dest.name}: content does not match its type (refused)")
            written += len(chunk)
            if written > info.file_size:
                raise OSError(f"{dest.name}: member is larger than its declared {info.file_size} bytes")
            out.write(chunk)
    tmp.replace(dest)


def layer_path(product_dir: Path, layer: str) -> Path | None:
    """The newest version of one GeoJSON layer of an extracted product, or ``None``."""
    return _layer(product_dir, layer)


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
    frame["source_time"] = [times.get(_source_id(s)) for s in frame.get("dmg_src_id", [None] * len(frame))]
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


def _source_id(value: Any) -> int | None:
    """``dmg_src_id`` as an int: GeoJSON may carry 2, 2.0 or "2"; a null stays ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number.is_integer() else None
