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
in ``P1-M0.3``, not to relax the constraint.

References
----------
``docs/04_data_access_and_sentinel1_preprocessing.md`` section 2.4.
``docs/00_mission_brief_cems.md`` section 3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["EventQuery", "MatchedScenes", "search_matched_pairs", "summarise_optical_availability"]


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
        The window during which the flood is believed to be in progress. Getting this
        wrong is a labelling error, not a query error: a post-event scene outside the
        window images a receded flood.
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


def search_matched_pairs(query: EventQuery, catalogue_url: str) -> MatchedScenes:
    """Search a STAC API for a matched pre/post S1 pair plus S2 and DEM coverage.

    Raises
    ------
    LookupError
        If no pre-event scene shares the post-event scene's relative orbit and pass
        direction. **Raise; do not fall back to a different orbit.**

    Notes
    -----
    Properties to filter on, all asserted by ``tests/test_stac_query.py``:

    ==================== ======================================================
    property             why
    ==================== ======================================================
    ``sat:relative_orbit`` the critical one -- must match between pre and post
    ``sat:orbit_state``    ascending / descending; must match
    ``sar:instrument_mode`` ``IW``
    ``sar:polarizations``  must contain both ``VV`` and ``VH``
    ``sar:product_type``   ``GRD``
    ``eo:cloud_cover``     S2 only; SCENE-level, not AOI-level
    ==================== ======================================================

    ``eo:cloud_cover`` is a **scene** statistic and is not the number you need. A scene
    that is 40 % cloudy may be completely clear over your AOI, or completely covered.
    Compute AOI-level cloud fraction from the SCL after reading
    (``docs/02`` section 6.3).

    On CDSE: tokens are short-lived, so refresh them programmatically rather than
    pasting one into a config; and concurrent-download quotas are enforced per account,
    so a bulk ingest spawning twenty parallel downloads is throttled rather than fast.
    ``P1-M1`` assumes a serial-with-retry ingest.

    TODO(student)
    -------------
    1. Search S1 GRD in the flood window; select the post-event scene.
    2. Search backwards ``pre_event_window_days`` **filtered to that scene's relative
       orbit and orbit state**. Raise ``LookupError`` if empty.
    3. Search S2 L2A across the window, keeping the cloud cover for each.
    4. Search the DEM collection for tiles covering the bbox.
    5. Return a frozen :class:`MatchedScenes`. Persist it; ``P1-M0``'s acceptance
       criterion is that the query is re-runnable and reproducible.
    """
    raise NotImplementedError("Implement the STAC query. See docs/04 section 2.4.")


def summarise_optical_availability(matches: tuple[MatchedScenes, ...]) -> dict[str, float]:
    r"""Compute the empirical :math:`p_{\mathrm{clear}}` that justifies the architecture.

    Returns
    -------
    dict[str, float]
        Per event and overall: overpasses, usable overpasses, fraction usable.

    Notes
    -----
    This is ``P1-M0`` exercise 2 and it is **the** numerical justification for building
    a fusion model at all. An optical observation is usable only if *all* of: an
    overpass occurs in the flood window; the flood is still in progress; the pixel is
    not cloud; the pixel is not cloud *shadow* (a larger area than the cloud at oblique
    sun angles); and it is daylight. Conditions 2, 4 and 5 are usually forgotten and
    they are multiplicative.

    The binomial expression :math:`1 - (1-p)^N` of ``docs/00`` section 4.3 is
    **optimistic**: cloud is strongly autocorrelated in time, so the effective number of
    independent trials is far below :math:`N`; and the probability of cloud *given a
    flood* is much higher than the climatological mean, because the same weather system
    causes both. Quoting an annual average cloud fraction therefore systematically
    overstates availability.

    Expect a disappointing result and **report it anyway**. If across four events you
    find two usable optical scenes, that is the finding -- and it is precisely what makes
    ablation row 11 the operationally meaningful row rather than row 9.

    TODO(student)
    -------------
    1. Count overpasses per event and AOI-level usable fraction from the SCL.
    2. Compute :math:`p_{\mathrm{clear}}`; compare with ``docs/00`` section 4.3.
    3. Write the two-sentence conclusion: does this justify fusion, or argue for
       SAR-only? Either is acceptable if supported.
    """
    raise NotImplementedError("Implement availability statistics. See docs/02 section 6.3.")
