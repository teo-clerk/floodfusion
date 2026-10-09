r"""STAC access: finding matched Sentinel-1 / Sentinel-2 / DEM items.

Milestone ``P1-M0``. The deliverable is a **scripted, re-runnable** query, not a list of
scene IDs pasted into a notebook.

The one constraint that is not negotiable
-----------------------------------------
The pre-event Sentinel-1 scene must share the **relative orbit** and **pass direction**
of the post-event scene. This is a correctness condition, not a refinement
(``docs/04`` section 7):

- Two relative orbits can differ by 10 degrees or more in incidence angle, giving a
  systematic offset of ~1.2 dB over unchanged land -- and the offset is *land-cover
  dependent*, so no global correction removes it. What remains is a land-cover-correlated
  change signal, which is exactly the structure a CNN will learn as if it were flooding.
- Layover and shadow masks **move** with the geometry, producing apparent changes above
  10 dB -- larger than the flood signal, and located in valleys.
- Ascending versus descending inverts the look azimuth by ~180 degrees, so every slope
  that faced the sensor now faces away and the change map is pure topography.

A candidate event for which this cannot be satisfied is a reason to **drop the event**
in ``P1-M0.3``, not to relax the constraint: :func:`search_matched_pairs` raises
:class:`LookupError` and never falls back.

Catalogues
----------
================== ============================================== ======================
profile            collections (S1 / S2 / DEM)                    access
================== ============================================== ======================
planetary_computer ``sentinel-1-rtc`` / ``sentinel-2-l2a`` /      anonymous SAS signing
                   ``cop-dem-glo-30``                             (``planetary_computer``)
cdse               ``sentinel-1-grd`` / ``sentinel-2-l2a`` /      search anonymous;
                   ``cop-dem-glo-30-dged-cog``                    downloads need an account
================== ============================================== ======================

Property names differ between catalogues and are verified against live items: both use
``sat:relative_orbit``, ``sat:orbit_state``, ``sar:instrument_mode`` and
``sar:polarizations``; CDSE carries the product type in ``product:type``
(``IW_GRDH_1S``), Planetary Computer's RTC collection needs no product-type filter.

Torch-free; ``pystac_client`` and ``planetary_computer`` are imported only when a live
client is opened. :class:`InMemoryCatalogue` gives the same search interface offline (tests,
and replaying a persisted query).

References
----------
``docs/04_data_access_and_sentinel1_preprocessing.md`` section 2.4.
``docs/00_mission_brief_cems.md`` section 3.1.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

__all__ = [
    "PROFILES",
    "CatalogueProfile",
    "EventQuery",
    "InMemoryCatalogue",
    "MatchedScenes",
    "load_matched_scenes",
    "open_catalogue",
    "profile_for",
    "redact",
    "save_matched_scenes",
    "search_matched_pairs",
    "summarise_optical_availability",
]

LOGGER = logging.getLogger(__name__)

#: Two frames of one Sentinel-1 datatake are ~25 s apart; scenes closer than this are
#: one acquisition (mosaicked together), farther apart they are different passes.
_SAME_ACQUISITION = timedelta(minutes=3)


@dataclass(frozen=True)
class EventQuery:
    """One flood event to search for.

    Attributes
    ----------
    event_id
        Stable identifier used as a manifest key.
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    flood_start, flood_end
        The window during which the flood is believed to be in progress, **inclusive at
        day resolution**: activations are reported by date, so ``flood_end`` of
        2023-11-07 includes the 17:07 pass of that day. Getting the window wrong is a
        labelling error, not a query error: a post-event scene outside it images a
        receded flood.
    pre_event_window_days
        How far back to look for the reference acquisition. Longer windows find more
        candidates but risk seasonal vegetation change confounding the difference
        (``docs/01`` section 10.3).
    """

    event_id: str
    bbox: tuple[float, float, float, float]
    flood_start: datetime
    flood_end: datetime
    pre_event_window_days: int = 60


@dataclass(frozen=True)
class MatchedScenes:
    """The result of a successful match for one event."""

    event_id: str
    s1_post_id: str
    s1_pre_id: str
    relative_orbit: int
    orbit_state: str
    s2_ids: tuple[str, ...]
    s2_cloud_cover: tuple[float, ...]
    dem_ids: tuple[str, ...]
    #: Every frame of the post/pre acquisition intersecting the AOI (mosaicked at ingest).
    s1_post_frames: tuple[str, ...] = ()
    s1_pre_frames: tuple[str, ...] = ()
    s1_post_datetime: str = ""
    s1_pre_datetime: str = ""
    s2_datetimes: tuple[str, ...] = ()
    catalogue: str = ""


@dataclass(frozen=True)
class CatalogueProfile:
    """Collection ids and property conventions of one STAC API."""

    name: str
    s1_collection: str
    s2_collection: str
    dem_collection: str | None
    product_type_key: str | None = None
    product_type_prefix: str | None = None
    sign_with_planetary_computer: bool = False


PROFILES: dict[str, CatalogueProfile] = {
    "planetary_computer": CatalogueProfile(
        "planetary_computer", "sentinel-1-rtc", "sentinel-2-l2a", "cop-dem-glo-30",
        sign_with_planetary_computer=True,
    ),
    "cdse": CatalogueProfile(
        "cdse", "sentinel-1-grd", "sentinel-2-l2a", "cop-dem-glo-30-dged-cog",
        product_type_key="product:type", product_type_prefix="IW_GRD",
    ),
    "generic": CatalogueProfile(
        "generic", "sentinel-1-grd", "sentinel-2-l2a", "cop-dem-glo-30",
        product_type_key="sar:product_type", product_type_prefix="GRD",
    ),
}


#: Exact catalogue hosts (never a substring test: ``planetarycomputer.evil.example`` and
#: ``planetarycomputer.microsoft.com@evil.example`` must not pass as Planetary Computer).
_KNOWN_HOSTS = {"planetarycomputer.microsoft.com": "planetary_computer",
                "catalogue.dataspace.copernicus.eu": "cdse"}
_HTTP_TIMEOUT = (10.0, 60.0)


def profile_for(catalogue_url: str) -> CatalogueProfile:
    """Pick the profile from the catalogue's exact https host; anything else is ``generic``."""
    parsed = urlparse(catalogue_url)
    name = _KNOWN_HOSTS.get((parsed.hostname or "").lower()) if parsed.scheme == "https" else None
    return PROFILES[name or "generic"]


def redact(text: str) -> str:
    """Strip query strings (SAS tokens) from every URL in ``text`` — for logs and errors."""
    import re

    return re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\"]*", r"\1?<redacted>", str(text))


class _Searchable(Protocol):
    def search(self, **kwargs: Any) -> Any: ...


def open_catalogue(catalogue_url: str, profile: CatalogueProfile | None = None) -> Any:
    """Open a live STAC API client (``pystac_client``), signing Planetary Computer hrefs."""
    from pystac_client import Client

    profile = profile or profile_for(catalogue_url)
    modifier = None
    if profile.sign_with_planetary_computer:
        import planetary_computer

        modifier = planetary_computer.sign_inplace
    return Client.open(catalogue_url, modifier=modifier, timeout=_HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# Item accessors: pystac.Item and plain STAC-JSON dicts alike
# ---------------------------------------------------------------------------
def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if to_dict is None:
        raise TypeError(f"not a STAC item: {type(item).__name__}")
    return to_dict(include_self_link=False, transform_hrefs=False)


def _datetime_of(item: Mapping[str, Any]) -> datetime:
    props = item.get("properties", {})
    text = props.get("datetime") or props.get("start_datetime")
    if not text:
        raise ValueError(f"item {item.get('id')} has no datetime")
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _bbox_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def _day_window(query: EventQuery) -> tuple[datetime, datetime]:
    start = datetime.combine(query.flood_start.date(), time.min, tzinfo=UTC)
    end = datetime.combine(query.flood_end.date(), time.max, tzinfo=UTC)
    if end < start:
        raise ValueError(f"{query.event_id}: flood_end precedes flood_start")
    return start, end


def _interval(start: datetime, end: datetime) -> str:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return f"{start.astimezone(UTC).strftime(fmt)}/{end.astimezone(UTC).strftime(fmt)}"


def _search(client: _Searchable, collection: str, bbox: Sequence[float], interval: str | None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"collections": [collection], "bbox": list(bbox), "max_items": 500}
    if interval is not None:
        kwargs["datetime"] = interval
    result = client.search(**kwargs)
    items = result.items() if hasattr(result, "items") else result
    return [_as_dict(i) for i in items]


def _is_usable_s1(item: Mapping[str, Any], profile: CatalogueProfile) -> bool:
    props = item.get("properties", {})
    mode = props.get("sar:instrument_mode")
    if mode is not None and str(mode).upper() != "IW":
        return False
    pols = {str(p).upper() for p in props.get("sar:polarizations", [])}
    if pols and not {"VV", "VH"} <= pols:
        return False
    product_type = props.get(profile.product_type_key) if profile.product_type_key else None
    if product_type is not None and not str(product_type).upper().startswith(profile.product_type_prefix or ""):
        return False
    return props.get("sat:relative_orbit") is not None and props.get("sat:orbit_state") is not None


def _group_acquisitions(items: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group frames into acquisitions: same orbit and pass, datetimes within minutes."""
    ordered = sorted(items, key=_datetime_of)
    groups: list[list[dict[str, Any]]] = []
    for item in ordered:
        props = item["properties"]
        key = (props["sat:relative_orbit"], props["sat:orbit_state"])
        last = groups[-1] if groups else None
        if last is not None:
            lp = last[-1]["properties"]
            same = (lp["sat:relative_orbit"], lp["sat:orbit_state"]) == key
            if same and _datetime_of(item) - _datetime_of(last[-1]) <= _SAME_ACQUISITION:
                last.append(item)
                continue
        groups.append([item])
    return groups


def _s2_dedupe_key(item: Mapping[str, Any]) -> tuple[str, str]:
    """(MGRS tile, sensing time): one physical acquisition of one tile.

    Planetary Computer serves reprocessed duplicates — the same tile and sensing time under
    processing baselines 05.09 (near-real-time) and 05.10 (2024 reprocessing), with
    different item ids — which would double-count overpasses and mix baselines in a mosaic.
    """
    props = item.get("properties", {})
    tile = str(props.get("s2:mgrs_tile") or props.get("mgrs:utm_zone", ""))
    if not tile:
        parts = str(item.get("id", "")).split("_")
        tile = next((p for p in parts if len(p) == 6 and p.startswith("T")), str(item.get("id")))
    return tile, _datetime_of(item).strftime("%Y%m%dT%H%M%S")


def _dedupe_s2(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest processing baseline (then the latest-processed id) per tile and time."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = _s2_dedupe_key(item)
        rank = (str(item.get("properties", {}).get("s2:processing_baseline", "")), str(item.get("id")))
        current = best.get(key)
        if current is None or rank > (str(current.get("properties", {}).get("s2:processing_baseline", "")),
                                       str(current.get("id"))):
            best[key] = item
    return list(best.values())


def _primary_frame(group: Sequence[dict[str, Any]], bbox: Sequence[float]) -> dict[str, Any]:
    return max(group, key=lambda it: (_bbox_overlap(it.get("bbox", bbox), bbox), it["id"]))


def search_matched_pairs(
    query: EventQuery,
    catalogue_url: str,
    *,
    client: _Searchable | None = None,
    profile: CatalogueProfile | None = None,
) -> MatchedScenes:
    """Search a STAC API for a matched pre/post S1 pair plus S2 and DEM coverage.

    Parameters
    ----------
    query
        The event.
    catalogue_url
        STAC API root; selects the :data:`PROFILES` entry by host.
    client
        Any object with a ``pystac_client``-compatible ``search`` — a live client by
        default, an :class:`InMemoryCatalogue` offline.

    Raises
    ------
    LookupError
        If there is no post-event IW VV+VH scene in the flood window, or if **no
        pre-event scene shares the post-event scene's relative orbit and pass
        direction**. Raise; never fall back to a different orbit.

    Notes
    -----
    Selection:

    1. **Post-event**: the earliest IW acquisition with both VV and VH in the
       (day-inclusive) flood window — the flood is most extensive early. All frames of
       that acquisition intersecting the AOI are kept for mosaicking.
    2. **Pre-event**: the latest acquisition in the preceding ``pre_event_window_days`` on
       **the same relative orbit and orbit state** — closest in time, so vegetation
       phenology changes least (``docs/01`` section 10.3).
    3. **Sentinel-2 L2A** in the flood window, with the scene-level ``eo:cloud_cover``.
       That is a *scene* statistic, not the AOI-level number the model needs; the AOI
       cloud fraction is computed from the SCL at ingest (``docs/02`` section 6.3).
       Reprocessed duplicates (same tile and sensing time, newer baseline) are collapsed
       to the highest processing baseline.
    4. **DEM tiles** intersecting the AOI.
    """
    profile = profile or profile_for(catalogue_url)
    client = client or open_catalogue(catalogue_url, profile)
    start, end = _day_window(query)

    post_items = [i for i in _search(client, profile.s1_collection, query.bbox, _interval(start, end))
                  if _is_usable_s1(i, profile)]
    if not post_items:
        raise LookupError(
            f"{query.event_id}: no Sentinel-1 IW VV+VH scene between {start.date()} and "
            f"{end.date()} in {profile.s1_collection}"
        )
    post_group = _group_acquisitions(post_items)[0]
    post = _primary_frame(post_group, query.bbox)
    rel = int(post["properties"]["sat:relative_orbit"])
    state = str(post["properties"]["sat:orbit_state"]).lower()

    pre_start = start - timedelta(days=query.pre_event_window_days)
    candidates = [
        i for i in _search(client, profile.s1_collection, query.bbox, _interval(pre_start, start - timedelta(seconds=1)))
        if _is_usable_s1(i, profile)
        and int(i["properties"]["sat:relative_orbit"]) == rel
        and str(i["properties"]["sat:orbit_state"]).lower() == state
    ]
    if not candidates:
        raise LookupError(
            f"{query.event_id}: no pre-event scene on relative orbit {rel} ({state}) within "
            f"{query.pre_event_window_days} days before {start.date()}. Drop the event — do "
            "not relax the constraint (docs/05 section 2.1, criterion 3)."
        )
    pre_group = _group_acquisitions(candidates)[-1]
    pre = _primary_frame(pre_group, query.bbox)

    s2_items = sorted(
        _dedupe_s2(_search(client, profile.s2_collection, query.bbox, _interval(start, end))),
        key=lambda i: (_datetime_of(i), i["id"]),
    )
    dem_ids: tuple[str, ...] = ()
    if profile.dem_collection:
        try:
            dem_ids = tuple(sorted(i["id"] for i in _search(client, profile.dem_collection, query.bbox, None)))
        except Exception as exc:  # noqa: BLE001 - a missing DEM listing must not hide the S1 match
            LOGGER.warning("%s: DEM search in %s failed (%s); dem_ids left empty",
                           query.event_id, profile.dem_collection, exc)
    return MatchedScenes(
        event_id=query.event_id,
        s1_post_id=post["id"],
        s1_pre_id=pre["id"],
        relative_orbit=rel,
        orbit_state=state,
        s2_ids=tuple(i["id"] for i in s2_items),
        s2_cloud_cover=tuple(float(i["properties"].get("eo:cloud_cover", math.nan)) for i in s2_items),
        dem_ids=dem_ids,
        s1_post_frames=tuple(sorted(i["id"] for i in post_group)),
        s1_pre_frames=tuple(sorted(i["id"] for i in pre_group)),
        s1_post_datetime=_datetime_of(post).isoformat(),
        s1_pre_datetime=_datetime_of(pre).isoformat(),
        s2_datetimes=tuple(_datetime_of(i).isoformat() for i in s2_items),
        catalogue=profile.name,
    )


def summarise_optical_availability(
    matches: tuple[MatchedScenes, ...],
    *,
    cloud_threshold_percent: float = 20.0,
) -> dict[str, Any]:
    r"""Empirical :math:`p_{\mathrm{clear}}`: the number that justifies (or not) fusion.

    Returns
    -------
    dict
        ``n_events``, ``overpasses``, ``usable`` (scene cloud cover below the threshold),
        ``p_clear`` (``nan`` when there are no overpasses), ``per_event`` breakdown, and
        ``n_for_90pct`` — the binomial :math:`\lceil\ln 0.1/\ln(1-p)\rceil` overpasses for
        90 % confidence of one usable scene.

    Notes
    -----
    This uses the **scene-level** ``eo:cloud_cover``; the AOI-level usable fraction is
    computed from the SCL at ingest, and both are optimistic: a usable overpass also needs
    the flood still in progress, no cloud *shadow* and daylight — conditions that
    multiply. And the binomial expression assumes independent passes, while cloud is
    autocorrelated in time and correlated with the flood itself (``docs/00`` section 4.3),
    so the true availability is lower still. Report the disappointing number anyway.
    """
    per_event: dict[str, dict[str, float]] = {}
    overpasses = usable = 0
    for m in matches:
        passes = _overpass_cloud_cover(m)
        n = len(passes)
        u = sum(1 for c in passes if c == c and c < cloud_threshold_percent)
        per_event[m.event_id] = {"overpasses": n, "usable": u, "fraction_usable": (u / n) if n else math.nan}
        overpasses += n
        usable += u
    p_clear = usable / overpasses if overpasses else math.nan
    if p_clear != p_clear or p_clear <= 0.0:
        n90 = math.inf
    elif p_clear >= 1.0:
        n90 = 1
    else:
        n90 = math.ceil(math.log(0.1) / math.log(1.0 - p_clear))
    return {
        "n_events": len(matches),
        "overpasses": overpasses,
        "usable": usable,
        "p_clear": p_clear,
        "cloud_threshold_percent": cloud_threshold_percent,
        "n_for_90pct": n90,
        "per_event": per_event,
    }


def _overpass_cloud_cover(m: MatchedScenes) -> list[float]:
    """Scene cloud cover per **overpass**: tiles sensed within the same minute are one pass.

    An AOI on a tile boundary is covered by two items per pass; counting items would double
    the overpasses. The pass's cloud cover is the mean over its tiles. Without datetimes,
    every entry is taken as its own pass.
    """
    if len(m.s2_datetimes) != len(m.s2_cloud_cover):
        return list(m.s2_cloud_cover)
    groups: dict[str, list[float]] = {}
    for when, cover in zip(m.s2_datetimes, m.s2_cloud_cover):
        groups.setdefault(str(when)[:16], []).append(cover)
    return [float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
            for v in (np.asarray(g, dtype=np.float64) for g in groups.values())]


# ---------------------------------------------------------------------------
# Offline catalogue and persistence
# ---------------------------------------------------------------------------
@dataclass
class InMemoryCatalogue:
    """A STAC search over a fixed list of item dicts — tests, and replaying a saved query.

    ``search(collections=..., ids=..., bbox=..., datetime="start/end")`` filters by
    collection, item id, bbox intersection and the (inclusive) datetime interval, like a
    STAC API.
    """

    items: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, items: Iterable[Mapping[str, Any]]) -> InMemoryCatalogue:
        return cls([dict(i) for i in items])

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        collections = set(kwargs.get("collections") or [])
        ids = set(kwargs.get("ids") or [])
        bbox = kwargs.get("bbox")
        interval = kwargs.get("datetime")
        lo = hi = None
        if interval:
            a, _, b = str(interval).partition("/")
            lo = datetime.fromisoformat(a.replace("Z", "+00:00")) if a and a != ".." else None
            hi = datetime.fromisoformat(b.replace("Z", "+00:00")) if b and b != ".." else None
        out = []
        for item in self.items:
            if collections and item.get("collection") not in collections:
                continue
            if ids and item.get("id") not in ids:
                continue
            if bbox is not None and "bbox" in item and _bbox_overlap(item["bbox"], bbox) <= 0.0:
                continue
            if lo is not None or hi is not None:
                dt = _datetime_of(item)
                if (lo is not None and dt < lo) or (hi is not None and dt > hi):
                    continue
            out.append(item)
        return out

    def save(self, path: Path | str) -> None:
        """Persist the items with every href's query string (a signing token) removed."""
        def strip(item: Mapping[str, Any]) -> dict[str, Any]:
            assets = {k: {**v, "href": str(v.get("href", "")).split("?", 1)[0]}
                      for k, v in dict(item.get("assets", {})).items()}
            return {**item, "assets": assets}

        Path(path).write_text(json.dumps({"items": [strip(i) for i in self.items]}, indent=1, default=str),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> InMemoryCatalogue:
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["items"])


def save_matched_scenes(matches: Sequence[MatchedScenes], path: Path | str) -> None:
    """Persist query results: ``P1-M0`` requires the query to be re-runnable and recorded."""
    Path(path).write_text(json.dumps([asdict(m) for m in matches], indent=2), encoding="utf-8")


def load_matched_scenes(path: Path | str) -> tuple[MatchedScenes, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(MatchedScenes(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in m.items()}) for m in raw)
