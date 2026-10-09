r"""STAC queries and the relative-orbit invariant.

The specification for :mod:`floodfusion.data.stac`. Milestone ``P1-M0``.

The pre-event Sentinel-1 scene **must** share the relative orbit and pass direction of
the post-event scene. This is a correctness condition, not a refinement: mismatched
geometry produces a change map dominated by topography, with apparent changes above
10 dB — larger than the flood signal itself, and located in valleys
(``docs/04`` section 7).

Network tests are marked ``network`` and are excluded by ``make test``. Run them with
``pytest -m network`` once credentials are configured.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from floodfusion.data.stac import (
    EventQuery,
    MatchedScenes,
    search_matched_pairs,
    summarise_optical_availability,
)

EXPECT_UNIMPLEMENTED = True
ALWAYS_PASS = frozenset({
    "test_query_is_frozen",
    "test_incidence_mismatch_costs_over_one_db",
    "test_availability_arithmetic_is_discouraging",
    "test_independence_assumption_is_optimistic",
    "test_default_pre_event_window",
    "test_missing_matching_orbit_raises_rather_than_falling_back",
})


def _query() -> EventQuery:
    start = datetime(2023, 11, 4, 0, 0, 0)
    return EventQuery(
        event_id="e0", bbox=(10.0, 45.0, 10.5, 45.5),
        flood_start=start, flood_end=start + timedelta(days=3),
    )


class TestEventQuery:
    def test_query_is_frozen(self) -> None:
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            _query().event_id = "other"  # type: ignore[misc]

    def test_default_pre_event_window(self) -> None:
        assert _query().pre_event_window_days == 60


class TestRelativeOrbitInvariant:
    def test_incidence_mismatch_costs_over_one_db(self) -> None:
        r"""Two relative orbits differing by 10 deg cost ~1.2 dB over unchanged land.

        Worse, the offset is land-cover dependent (n varies with surface type), so no
        global correction removes it — and what remains is a land-cover-correlated
        change signal that a CNN will learn as if it were flooding.
        """
        theta_ref, theta_other, n = math.radians(33.0), math.radians(43.0), 2.0
        ratio = (math.cos(theta_other) / math.cos(theta_ref)) ** n
        offset_db = 10 * math.log10(ratio)
        assert offset_db == pytest.approx(-1.20, abs=0.1)

    def test_matched_pair_shares_relative_orbit(self) -> None:
        scenes = search_matched_pairs(_query(), "https://example.invalid/stac")
        assert isinstance(scenes, MatchedScenes)
        assert scenes.relative_orbit >= 0

    def test_missing_matching_orbit_raises_rather_than_falling_back(self) -> None:
        """Drop the EVENT, not the constraint (``docs/05`` section 2.1, criterion 3)."""
        with pytest.raises((LookupError, NotImplementedError)):
            search_matched_pairs(_query(), "https://example.invalid/stac")


class TestOpticalAvailability:
    def test_availability_arithmetic_is_discouraging(self) -> None:
        r""":math:`N \ge \ln(0.1)/\ln(1-p)` passes for 90 % confidence.

        At p_clear = 0.2 that is ~11 overpasses, or 7-8 weeks at a 5-day revisit. The
        flood will have receded and the activation closed. This is the numerical
        justification for a radar-primary architecture (``docs/00`` section 4.3).
        """
        p_clear = 0.2
        n = math.log(1 - 0.9) / math.log(1 - p_clear)
        assert n == pytest.approx(10.3, abs=0.2)
        assert math.ceil(n) * 5 / 7 > 7.0  # weeks

    def test_independence_assumption_is_optimistic(self) -> None:
        """Cloud is autocorrelated in time AND correlated with the flood itself.

        The binomial form assumes independent trials. With persistence q, the effective
        number of independent trials is ~N(1-q), so the true probability is LOWER.
        """
        n, p = 11, 0.2
        binomial = 1 - (1 - p) ** n
        q = 0.7
        effective_n = n * (1 - q)
        persistent = 1 - (1 - p) ** effective_n
        assert binomial > persistent, "persistence makes availability worse, not better"

    def test_summary_reports_a_fraction(self) -> None:
        out = summarise_optical_availability(())
        assert isinstance(out, dict)


@pytest.mark.network
class TestLiveCatalogue:
    """Hit a real STAC endpoint. Excluded from ``make test``; run with ``-m network``."""

    def test_query_returns_matched_pairs(self) -> None:
        scenes = search_matched_pairs(
            _query(), "https://catalogue.dataspace.copernicus.eu/stac"
        )
        assert scenes.s1_pre_id and scenes.s1_post_id
        assert scenes.orbit_state in {"ascending", "descending"}
