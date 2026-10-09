"""Archive access for the real-data ingest: catalogue client, byte fetcher, item lookups.

Everything here is injectable so the ingest runs offline against local files and an
:class:`floodfusion.data.stac.InMemoryCatalogue` (``tests/test_ingest_offline.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodfusion.data.stac import open_catalogue, redact

__all__ = [
    "GDAL_ENV",
    "MAX_FETCH_BYTES",
    "PC_STAC_URL",
    "IngestSources",
    "as_dict",
    "grd_id_for_rtc",
    "items_by_id",
    "items_in_bbox",
    "product_annotation_href",
]

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_HTTP_TIMEOUT_S = 120
MAX_FETCH_BYTES = 64 * 1024 * 1024
#: GDAL's own HTTP reads (COGs on blob storage) need bounds too.
GDAL_ENV = {"GDAL_HTTP_TIMEOUT": 60, "GDAL_HTTP_MAX_RETRY": 3, "GDAL_HTTP_RETRY_DELAY": 2,
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}


def _default_fetch(href: str) -> bytes:
    """Bytes of a local file or an allow-listed, signed HTTPS href (no redirects, size-capped)."""
    from floodfusion.data.sources import check_href, signed

    href = check_href(href)
    if not href.startswith("https://"):
        path = Path(href.removeprefix("file://"))
        if not path.is_file() or path.stat().st_size > MAX_FETCH_BYTES:
            raise ValueError(f"{path}: not a regular file below {MAX_FETCH_BYTES} bytes")
        return path.read_bytes()
    import requests

    url = signed(href)
    with requests.get(url, timeout=(10, _HTTP_TIMEOUT_S), allow_redirects=False, stream=True) as response:
        if response.status_code != 200:
            raise OSError(f"HTTP {response.status_code} for {redact(url)}")
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=1 << 20):
            size += len(chunk)
            if size > MAX_FETCH_BYTES:
                raise OSError(f"{redact(url)}: larger than {MAX_FETCH_BYTES} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class IngestSources:
    """The catalogue client and the byte fetcher; both replaceable offline."""

    client: Any
    fetch_bytes: Callable[[str], bytes] = _default_fetch

    @classmethod
    def open(cls, catalogue_url: str = PC_STAC_URL) -> IngestSources:
        return cls(client=open_catalogue(catalogue_url))


def as_dict(item: Any) -> dict[str, Any]:
    return dict(item) if isinstance(item, Mapping) else item.to_dict(include_self_link=False, transform_hrefs=False)


def items_by_id(client: Any, collection: str, ids: Sequence[str]) -> list[dict[str, Any]]:
    """The items with these ids, in the order given; ``LookupError`` if any is missing."""
    if not ids:
        return []
    result = client.search(collections=[collection], ids=list(ids), max_items=len(ids) + 5)
    found = {d["id"]: d for d in (as_dict(i) for i in (result.items() if hasattr(result, "items") else result))}
    missing = [i for i in ids if i not in found]
    if missing:
        raise LookupError(f"items not found in {collection}: {missing}")
    return [found[i] for i in ids]


def items_in_bbox(client: Any, collection: str, bbox: Sequence[float]) -> list[dict[str, Any]]:
    result = client.search(collections=[collection], bbox=list(bbox), max_items=200)
    return [as_dict(i) for i in (result.items() if hasattr(result, "items") else result)]


def grd_id_for_rtc(rtc_id: str) -> str:
    """Planetary Computer names an RTC item after its source GRD product plus ``_rtc``."""
    return rtc_id[:-4] if rtc_id.endswith("_rtc") else rtc_id


def product_annotation_href(grd_item: Mapping[str, Any], polarisation: str = "vv") -> str:
    """Href of the GRD **product** annotation (``annotation/<...>.xml``).

    Planetary Computer's ``schema-product-<pol>`` asset points at the **RFI** annotation
    (``annotation/rfi/rfi-iw-<pol>.xml``), not the product annotation; the product
    annotation sits one directory up as ``annotation/iw-<pol>.xml`` (verified on
    S1A_IW_GRDH_1SDV_20231107T170700_..._062A12).
    """
    assets = grd_item.get("assets", {})
    pol = polarisation.lower()
    for key in (f"annotation-{pol}", f"product-annotation-{pol}"):
        if key in assets:
            return str(assets[key]["href"])
    key = f"schema-product-{pol}"
    if key not in assets:
        raise KeyError(f"GRD item {grd_item.get('id')} has no {key} asset")
    href = str(assets[key]["href"])
    base, sep, query = href.partition("?")
    if "/annotation/rfi/rfi-" in base:
        base = base.replace("/annotation/rfi/rfi-", "/annotation/")
    return base + sep + query
