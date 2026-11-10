"""v1.3's ladder data: the CEMS benchmark's pairs as one root, regrouped by place (protocol v1.3 §2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodfusion.data import pool
from floodfusion.data.records import ChipRecord, load_manifest, load_manifest_meta, write_manifest


def _pair(root: Path, key: str, stratum: str, x: float, y: float, n_chips: int = 2) -> None:
    """One benchmark pair: pair.json and one ingested tile (UTM 33N, 5 km square) with n chips."""
    activation = key.split("_")[0]
    (root / key).mkdir(parents=True)
    (root / key / "pair.json").write_text(json.dumps({"key": key, "activation": activation, "stratum": stratum}))
    event = root / key / "events" / f"{key}_t00_00"
    event.mkdir(parents=True)
    (event / "stack.tif").write_bytes(b"")
    (event / "event.json").write_text(json.dumps({"event_id": event.name, "epsg": 32633, "width": 500, "height": 500,
                                                  "transform": [10, 0, x, 0, -10, y]}))
    write_manifest([ChipRecord(chip_id=f"{event.name}_{i}", basin_id="AOI", event_id=event.name,
                               raster_path=event / "stack.tif", window=(0, 0, 256, 256), positive_fraction=0.1,
                               valid_fraction=1.0, optical_available=False, relative_orbit=1, orbit_state="ascending")
                    for i in range(n_chips)], event / "chips.parquet")


@pytest.fixture
def benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "cems_benchmark_v12"
    _pair(root, "EMSR700_AOI01", "sar", 500_000, 4_500_000)
    _pair(root, "EMSR700_AOI02", "optical", 520_000, 4_500_000)       # same activation: one event, one group
    _pair(root, "EMSR701_AOI01", "sar", 900_000, 6_000_000)
    _pair(root, "EMSR702_AOI01", "mixed", 300_000, 5_000_000)
    _pair(root, "EMSR664_AOI01", "sar", 300_000, 7_000_000)           # Forlì: excluded by default
    (root / "cems").mkdir()                                            # the product cache: not a pair
    return root


def test_a_cems_stack_belongs_to_its_activation() -> None:
    assert pool.source_event({"event_id": "EMSR657_AOI01_t00_00"}) == "cems:EMSR657"
    with pytest.raises(KeyError, match="names no source event"):
        pool.source_event({"event_id": "EMSR657"})


def test_collect_cems_lists_every_pair_but_the_excluded_ones(benchmark: Path, tmp_path: Path) -> None:
    out = tmp_path / "pool_v13" / "cems"
    pool.collect_cems(benchmark, out)
    manifest = out / "chips" / "manifest.parquet"
    records = load_manifest(manifest)
    assert sorted({r.event_id.rsplit("_", 2)[0] for r in records}) == ["EMSR700_AOI01", "EMSR700_AOI02",
                                                                       "EMSR701_AOI01", "EMSR702_AOI01"]
    assert all(Path(r.raster_path).is_file() for r in records)
    provenance = load_manifest_meta(manifest)["provenance"]
    assert provenance["strata"] == {"EMSR700_AOI01": "sar", "EMSR700_AOI02": "optical", "EMSR701_AOI01": "sar",
                                    "EMSR702_AOI01": "mixed"}
    assert provenance["excluded"] == ["EMSR664"]


def test_the_collected_root_pools_into_groups_by_activation_and_place(benchmark: Path, tmp_path: Path) -> None:
    for i in range(4):     # four more far-apart activations, so EMSR700's two pairs hold exactly a quarter
        _pair(benchmark, f"EMSR{703 + i}_AOI01", "sar", 300_000, 5_500_000 + 400_000 * i)
    pool.collect_cems(benchmark, tmp_path / "pool_v13" / "cems")
    summary = pool.build_pool([tmp_path / "pool_v13" / "cems"], tmp_path / "pool_v13")
    group = {r.event_id: r.basin_id for r in load_manifest(tmp_path / "pool_v13" / "chips" / "manifest.parquet")}
    assert group["EMSR700_AOI01_t00_00"] == group["EMSR700_AOI02_t00_00"]
    assert summary["n_events"] == 7 and summary["n_groups"] == 7
    assert summary["largest_share"] == pytest.approx(0.25)


def test_collect_cems_refuses_a_blind_activation(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _pair(root, "EMSR899_AOI01", "sar", 500_000, 4_500_000)
    with pytest.raises(ValueError, match="blind"):
        pool.collect_cems(root, tmp_path / "out")
