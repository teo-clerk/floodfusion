r"""Sentinel-1 acquisitions over an area: frames merged into passes, and how much of the area each covers.

Planetary Computer lists ``sentinel-1-rtc`` per **frame**; one pass over an AOI is usually two or
three consecutive frames on the same relative orbit, seconds apart. Both the CEMS benchmark
(which pass is closest to a delineation) and Kuro Siwo (which pass the annotators drew on) need
passes, not frames, and the share of an area a pass actually imaged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = ["Acquisition", "acquisitions", "best_covering", "coverage"]

LOGGER = logging.getLogger(__name__)

#: Frames of one acquisition are at most this far apart in time.
_ACQUISITION_GAP_S = 180


@dataclass(frozen=True)
class Acquisition:
    """One Sentinel-1 acquisition: its frames merged."""

    when: datetime
    relative_orbit: int
    orbit_state: str
    footprint: Any          # shapely geometry, EPSG:4326
    frame_ids: tuple[str, ...]


def acquisitions(client: Any, geometry: Any, start: datetime, end: datetime, *,
                 sleep: Callable[[float], None] | None = None) -> list[Acquisition]:
    """Every Sentinel-1 RTC acquisition intersecting ``geometry`` between ``start`` and ``end``.

    A rate-limited search is retried with a growing pause (:data:`floodfusion.data.stac.RATE_LIMIT_BACKOFF_S`).
    """
    import time

    from shapely.geometry import shape
    from shapely.ops import unary_union

    from floodfusion.data.stac import datetime_of, search_with_backoff

    items = search_with_backoff(client, sleep=sleep or time.sleep, collections=["sentinel-1-rtc"],
                                bbox=list(geometry.bounds), max_items=500,
                                datetime=f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}")
    if len(items) >= 500:
        LOGGER.warning("sentinel-1-rtc search returned the 500-item cap: the list may be truncated")
    items = [i for i in items if i.get("geometry") and shape(i["geometry"]).intersects(geometry)
             and isinstance(i.get("properties", {}).get("sat:relative_orbit"), int)
             and i["properties"].get("sat:orbit_state")]
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=datetime_of):
        p = item["properties"]
        last = groups[-1][-1] if groups else None
        if last is not None and last["properties"]["sat:relative_orbit"] == p["sat:relative_orbit"] and \
                (datetime_of(item) - datetime_of(last)).total_seconds() <= _ACQUISITION_GAP_S:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [Acquisition(datetime_of(g[0]), int(g[0]["properties"]["sat:relative_orbit"]),
                        str(g[0]["properties"]["sat:orbit_state"]).lower(),
                        unary_union([shape(i["geometry"]) for i in g]), tuple(i["id"] for i in g)) for g in groups]


def coverage(area: Any, footprint: Any) -> float:
    """Fraction of ``area`` inside ``footprint``, by area in a local equal-area projection."""
    import shapely
    from pyproj import CRS, Transformer

    lon, lat = area.centroid.x, area.centroid.y
    laea = CRS.from_proj4(f"+proj=laea +lat_0={lat} +lon_0={lon} +units=m")
    to = Transformer.from_crs(4326, laea, always_xy=True)

    def project(geometry: Any) -> Any:
        return shapely.transform(geometry, lambda x, y: to.transform(x, y), interleaved=False)

    total = project(area).area
    return float(project(area.intersection(footprint)).area / total) if total > 0 else 0.0


def best_covering(candidates: Sequence[Acquisition], area: Any) -> tuple[Acquisition, float] | None:
    """The acquisition covering the largest share of ``area`` (ties: the earlier), with that share."""
    scored = [(coverage(area, a.footprint), a) for a in candidates]
    if not scored:
        return None
    share, best = max(scored, key=lambda sa: (round(sa[0], 6), -sa[1].when.timestamp()))
    return best, share
