"""What the live service may read and map before blind floods are switched on (protocol v1.3 §6).

The rule, fixed in ``reports/v1_3_deviations.md`` (2026-11-11) before this code:

- **Codes:** while ``blind_floods`` in ``configs/live.yaml`` is off, every activation code from EMSR899 on is refused
  before its record is requested (:func:`check_code`).
- **Dates:** a map refuses a pass date whose optical window (±2 days) reaches 20 July 2026, the first event date of
  the blind set (:func:`check_date`), since it could show a blind flood.
- **Records:** of an activation record only its code, name, category, event date and AOI extents are read, never a
  product (:func:`allowed_fields`).

The switch goes on only by a commit made after ``v1.3.0-freeze`` (or after the ladder decides that there is no v1.3
model), on the author's go. Torch-free.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = ["BLIND_EVENTS_START", "BLIND_FIRST_CODE", "LIVE_CONFIG", "ActivationSummary", "AoiExtent",
           "BlindFloodRefusedError", "allowed_fields", "blind_floods_enabled", "check_code", "check_date"]

#: The blind set (``reports/protocol_v1_2.md`` §2.2): every flood activation from this code on.
BLIND_FIRST_CODE = 899
#: Its first event date (``reports/protocol_v1_2.md``, Corrections).
BLIND_EVENTS_START = date(2026, 7, 20)
LIVE_CONFIG = Path("configs/live.yaml")
#: As in the benchmark (``floodfusion.cems_benchmark.OPTICAL_WINDOW_DAYS``): optical within ±2 days of the pass.
_OPTICAL_WINDOW_DAYS = 2
_CODE = re.compile(r"EMSR([0-9]{3,4})", re.ASCII)
_FLOOD = "flood"


class BlindFloodRefusedError(ValueError):
    """A request that would read or map a blind flood while the switch is off."""


@dataclass(frozen=True)
class AoiExtent:
    number: int
    name: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ActivationSummary:
    """The only fields of an activation record the live service reads."""

    code: str
    name: str
    category: str
    event_date: date
    aois: tuple[AoiExtent, ...]


def blind_floods_enabled(path: Path = LIVE_CONFIG) -> bool:
    """The committed switch; a missing file means off."""
    import yaml

    path = Path(path)
    if not path.is_file():
        return False
    value = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("blind_floods", False)
    if not isinstance(value, bool):
        raise ValueError(f"{path}: blind_floods must be true or false, not {value!r}")
    return value


def check_code(code: str, *, enabled: bool) -> str:
    """``code`` if the service may request its record."""
    match = _CODE.fullmatch(code)
    if match is None:
        raise ValueError(f"not a CEMS rapid-mapping activation code: {code[:40]!r}")
    if int(match.group(1)) >= BLIND_FIRST_CODE and not enabled:
        raise BlindFloodRefusedError(f"{code} is in the blind set (EMSR{BLIND_FIRST_CODE} on), and blind floods are off")
    return code


def check_date(when: date, *, enabled: bool, optical_window_days: int = _OPTICAL_WINDOW_DAYS) -> date:
    """``when`` if a map of that pass date cannot reach a blind flood's dates."""
    if not enabled and when + timedelta(days=optical_window_days) >= BLIND_EVENTS_START:
        raise BlindFloodRefusedError(
            f"a map of {when.isoformat()} (optical up to {optical_window_days} days later) reaches the blind floods, "
            f"whose events start on 20 July 2026; blind floods are off")
    return when


def _aoi(raw: Mapping[str, Any]) -> AoiExtent:
    from shapely import wkt

    west, south, east, north = wkt.loads(str(raw["extent"])).bounds
    return AoiExtent(number=int(raw.get("number") or 0), name=str(raw.get("name", "")),
                     bbox=(float(west), float(south), float(east), float(north)))


def allowed_fields(record: Mapping[str, Any], *, enabled: bool) -> ActivationSummary:
    """The record's code, name, category, event date and AOI extents; nothing else is read."""
    code = check_code(str(record.get("code", "")), enabled=enabled)
    category = str(record.get("category", ""))
    if category.strip().lower() != _FLOOD:
        raise ValueError(f"{code} is not a flood activation ({category[:40]!r})")
    event = datetime.fromisoformat(str(record["eventTime"]).replace("Z", "+00:00")).date()
    return ActivationSummary(code=code, name=str(record.get("name", "")), category=category, event_date=event,
                             aois=tuple(_aoi(a) for a in record.get("aois", [])))
