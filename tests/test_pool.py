"""The v1.2 training pool: one manifest over several roots, groups recomputed jointly by place."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodfusion.data import pool
from floodfusion.data.records import ChipRecord, load_manifest, write_manifest


def _root(tmp_path: Path, name: str, source: str, events: dict[str, tuple[float, float]], n_chips: int = 2) -> Path:
    """A dataset root: one stack per event (UTM 33N, 5 km square at the given easting/northing), n chips each."""
    root = tmp_path / name
    records = []
    for event, (x, y) in events.items():
        stack = root / "events" / f"{name}_{event}" / "stack.tif"
        stack.parent.mkdir(parents=True)
        stack.write_bytes(b"")
        (stack.parent / "event.json").write_text(json.dumps({
            "event_id": stack.parent.name, "epsg": 32633, "transform": [10, 0, x, 0, -10, y], "width": 500,
            "height": 500, source: {"event": event}}))
        for i in range(n_chips):
            records.append(ChipRecord(chip_id=f"{stack.parent.name}_{i}", basin_id=f"old_{event}",
                                      event_id=stack.parent.name, raster_path=stack, window=(0, 0, 256, 256),
                                      positive_fraction=0.1, valid_fraction=1.0, optical_available=False,
                                      relative_orbit=1, orbit_state="ascending"))
    write_manifest(records, root / "chips" / "manifest.parquet")
    return root


def test_events_near_each_other_across_datasets_share_a_group(tmp_path: Path) -> None:
    ks = _root(tmp_path, "ks", "kurosiwo", {"EMSR118_Spain": (500_000, 4_500_000), "EMSR999_Far": (900_000, 6_000_000)})
    far = {f"EMSR{161 + i}": (300_000 + 50_000 * (i % 2), 5_000_000 + 400_000 * i) for i in range(5)}
    gf = _root(tmp_path, "gf", "geoid_flood", {"EMSR160": (530_000, 4_520_000), **far})
    summary = pool.build_pool([ks, gf], tmp_path / "pool")
    records = load_manifest(tmp_path / "pool" / "chips" / "manifest.parquet")
    group = {r.event_id: r.basin_id for r in records}
    assert group["ks_EMSR118_Spain"] == group["gf_EMSR160"]             # 36 km apart: one group
    assert len({group["ks_EMSR999_Far"], group["gf_EMSR161"], group["ks_EMSR118_Spain"]}) == 3
    assert summary["n_events"] == 8 and summary["n_groups"] == 7 and summary["n_chips"] == 16
    assert summary["largest_share"] == pytest.approx(4 / 16)
    assert all(Path(r.raster_path).is_file() for r in records)          # relative paths still resolve
    assert all(r.basin_id.startswith(pool.GROUP_PREFIX) for r in records)


def test_a_dominant_group_is_refused(tmp_path: Path) -> None:
    ks = _root(tmp_path, "ks", "kurosiwo", {"A": (500_000, 4_500_000), "B": (510_000, 4_500_000)})
    gf = _root(tmp_path, "gf", "geoid_flood", {"C": (900_000, 6_000_000)}, n_chips=1)
    with pytest.raises(ValueError, match="buffered spatial blocks"):
        pool.build_pool([ks, gf], tmp_path / "pool")                    # A + B: 4 of 5 windows


def test_a_stack_without_a_source_event_is_an_error() -> None:
    with pytest.raises(KeyError, match="names no source event"):
        pool.source_event({"event_id": "x"})
    assert pool.source_event({"kurosiwo": {"event": "EMSR1_X", "aoi": "01"}}) == "kurosiwo:EMSR1_X"
    assert pool.source_event({"geoid_flood": {"event": "EMSR160"}}) == "geoid_flood:EMSR160"


def test_the_largest_group_may_hold_exactly_a_quarter_of_the_windows(tmp_path: Path) -> None:
    spread = {f"E{i}": (300_000, 5_000_000 + 400_000 * i) for i in range(4)}     # four groups of 2 windows
    ks = _root(tmp_path, "ks", "kurosiwo", spread)
    assert pool.build_pool([ks], tmp_path / "pool")["largest_share"] == pytest.approx(0.25)
    more = _root(tmp_path, "gf", "geoid_flood", {"F": (310_000, 5_000_000)}, n_chips=1)   # joins E0: 3 of 9
    with pytest.raises(ValueError, match="33%"):
        pool.build_pool([ks, more], tmp_path / "pool2")
