"""The live service's rule before the blind-flood switch (``reports/v1_3_deviations.md``, 2026-11-11; protocol §6).

These tests are the rule's enforcement: while the committed switch is off, no blind code is requested and no map may
reach a blind flood's dates. Switching blind floods on edits ``test_the_committed_switch_is_off``, with a comment
naming the author's go.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from floodfusion import live_rules

ROOT = Path(__file__).resolve().parents[1]

RECORD = {
    "code": "EMSR664", "name": "Flood in Italy", "category": "Flood", "eventTime": "2023-05-16T11:00:00",
    "reason": "anything else in the record is not read", "stats": {"anything": 1},
    "aois": [{"number": "1", "name": "Forlì", "extent": "POLYGON ((11.9 44.5, 12.4 44.5, 12.4 44.2, 11.9 44.2, 11.9 44.5))",
              "products": [{"type": "DEL", "downloadPath": "https://example.invalid/product.zip"}]}],
}


def test_the_committed_switch_is_off() -> None:
    assert live_rules.blind_floods_enabled(ROOT / "configs" / "live.yaml") is False


def test_a_missing_switch_is_off_and_a_malformed_one_is_refused(tmp_path: Path) -> None:
    assert live_rules.blind_floods_enabled(tmp_path / "absent.yaml") is False
    bad = tmp_path / "live.yaml"
    bad.write_text("blind_floods: 'yes'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="true or false"):
        live_rules.blind_floods_enabled(bad)


def test_the_blind_set_starts_where_the_v12_protocol_says() -> None:
    from floodfusion.data.pool import BLIND_FIRST_CODE
    from floodfusion.prospective import FIRST_CODE

    assert live_rules.BLIND_FIRST_CODE == FIRST_CODE == BLIND_FIRST_CODE == 899
    assert live_rules.BLIND_EVENTS_START == date(2026, 7, 20)


@pytest.mark.parametrize("code", ["EMSR899", "EMSR926", "EMSR1203"])
def test_a_blind_code_is_refused_while_the_switch_is_off(code: str) -> None:
    with pytest.raises(live_rules.BlindFloodRefusedError, match="blind"):
        live_rules.check_code(code, enabled=False)
    assert live_rules.check_code(code, enabled=True) == code


@pytest.mark.parametrize("code", ["EMSR656", "EMSR664", "EMSR898"])
def test_a_seen_code_is_allowed(code: str) -> None:
    assert live_rules.check_code(code, enabled=False) == code


@pytest.mark.parametrize("code", ["EMSR12", "emsr664", "EMSR664; rm", "LSR664"])
def test_a_malformed_code_is_refused(code: str) -> None:
    with pytest.raises(ValueError, match="not a CEMS"):
        live_rules.check_code(code, enabled=True)


def test_a_date_whose_optical_window_reaches_the_blind_events_is_refused() -> None:
    last_allowed = date(2026, 7, 17)                       # its optical window ends on 19 July
    assert live_rules.check_date(last_allowed, enabled=False) == last_allowed
    for when in (date(2026, 7, 18), date(2026, 7, 20), date(2026, 11, 11)):
        with pytest.raises(live_rules.BlindFloodRefusedError, match="20 July 2026"):
            live_rules.check_date(when, enabled=False)
    assert live_rules.check_date(date(2026, 11, 11), enabled=True) == date(2026, 11, 11)


def test_a_map_refuses_a_blind_date_before_any_tile_is_ingested(tmp_path: Path, monkeypatch) -> None:
    from floodfusion import live

    monkeypatch.setattr(live, "tiles_for", lambda bbox: pytest.fail("a tile was planned"))
    with pytest.raises(live_rules.BlindFloodRefusedError):
        live.map_area((12.0, 44.2, 12.1, 44.3), date(2026, 8, 1), tmp_path, predictor=None, name="x", cfg=None,
                      sources=None, device=None)
    assert not any(tmp_path.iterdir())


def test_only_the_allowed_fields_of_a_seen_record_are_read() -> None:
    seen = live_rules.allowed_fields(RECORD, enabled=False)
    assert seen == live_rules.ActivationSummary(
        code="EMSR664", name="Flood in Italy", category="Flood", event_date=date(2023, 5, 16),
        aois=(live_rules.AoiExtent(number=1, name="Forlì", bbox=(11.9, 44.2, 12.4, 44.5)),))
    assert "download" not in repr(seen).lower() and "reason" not in repr(seen)


def test_a_blind_record_is_refused_even_if_it_was_fetched() -> None:
    with pytest.raises(live_rules.BlindFloodRefusedError):
        live_rules.allowed_fields({**RECORD, "code": "EMSR926"}, enabled=False)


def test_a_record_that_is_not_a_flood_is_refused() -> None:
    with pytest.raises(ValueError, match="not a flood"):
        live_rules.allowed_fields({**RECORD, "category": "Wildfire"}, enabled=False)
