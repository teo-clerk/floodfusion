"""Encoding, synthetic events, chipping, manifest and splitting — the P1-M1 data path.

Uses the miniature benchmark (2 events x 2 basins, 128 px chips) so the whole chain —
render, encode, write stack, chip, manifest, splits, normalisation statistics — runs in a
couple of seconds.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from floodfusion.config import DataConfig, GridConfig, NormalizationStats, ValidityFlag
from floodfusion.data.chipping import enumerate_windows, extract_chips, read_event_stack
from floodfusion.data.encoding import (
    N_CHIP_BANDS,
    ChipBand,
    decode_physical,
    encode_stack,
    harmonise_s2_dn,
    mndwi_from_reflectance,
    normalise,
)
from floodfusion.data.records import ChipRecord, load_manifest, manifest_sha256, write_manifest
from floodfusion.data.synthetic.benchmark import TINY_EVENTS, build_benchmark
from floodfusion.data.synthetic.landscape import EventSpec, look_azimuth_for
from floodfusion.data.synthetic.scene import render_event
from floodfusion.splits import (
    assign_basins,
    compute_norm_stats,
    freeze_splits,
    load_splits,
    random_chip_split,
    test_pixels_covered_by_training,
)

GRID = GridConfig()
NORM = NormalizationStats()
TINY_DATA = DataConfig(chip_size=128, train_stride=64, eval_stride=128, min_valid_fraction=0.3,
                       basin_buffer_m=300.0, num_workers=0)


@pytest.fixture(scope="module")
def tiny_benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    return build_benchmark(root, specs=TINY_EVENTS, data_cfg=TINY_DATA, grid_cfg=GRID, norm=NORM)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def _layers(shape=(8, 10), *, optical=True):
    rng = np.random.default_rng(0)
    vv_post = rng.normal(-12, 3, shape)
    vv_post[0, 0] = np.nan
    return dict(
        vv_post_db=vv_post, vh_post_db=rng.normal(-19, 3, shape),
        vv_pre_db=rng.normal(-11, 3, shape), vh_pre_db=rng.normal(-18, 3, shape),
        reflectance=np.stack([np.full(shape, 0.05), np.full(shape, 0.02), np.full(shape, 0.01)]) if optical else None,
        slope_deg=np.full(shape, 4.5), hand_m=np.full(shape, 12.0),
        local_incidence_deg=np.full(shape, 37.25),
        label=(rng.random(shape) < 0.2).astype(float), validity=np.zeros(shape, np.uint8),
        optical_quality=np.full(shape, 0.8) if optical else None,
        landcover=np.full(shape, 40, np.uint8), grid_cfg=GRID, norm=NORM,
    )


class TestEncoding:
    def test_round_trip_within_quantisation(self) -> None:
        layers = _layers()
        stack = encode_stack(**layers)
        assert stack.dtype == np.int16 and stack.shape == (N_CHIP_BANDS, 8, 10)
        phys = decode_physical(stack, GRID, NORM)
        ok = np.isfinite(layers["vv_post_db"])
        assert np.allclose(phys.gamma0_db[0][ok], layers["vv_post_db"][ok], atol=0.005)
        assert np.allclose(phys.reflectance[0], 0.05, atol=1e-4)
        assert np.allclose(phys.hand_m, 12.0, rtol=1e-3)
        assert np.allclose(phys.local_incidence_deg, 37.25)

    def test_nodata_becomes_nan_and_invalid(self) -> None:
        phys = decode_physical(encode_stack(**_layers()), GRID, NORM)
        assert np.isnan(phys.gamma0_db[0, 0, 0])
        assert phys.validity[0, 0] & ValidityFlag.NODATA
        out = normalise(phys, NORM)
        assert not out["valid"][0, 0, 0]
        assert out["sar"][0, 0, 0] == 0.0, "invalid pixels are zero AFTER normalisation"

    def test_delta_is_derived_from_the_dates(self) -> None:
        layers = _layers()
        phys = decode_physical(encode_stack(**layers), GRID, NORM)
        expected = layers["vv_post_db"] - layers["vv_pre_db"]
        ok = np.isfinite(expected)
        assert np.allclose(phys.gamma0_db[4][ok], expected[ok], atol=0.01)

    def test_absent_optical_has_zero_quality(self) -> None:
        phys = decode_physical(encode_stack(**_layers(optical=False)), GRID, NORM)
        assert np.isnan(phys.reflectance).all() and (phys.optical_quality == 0).all()
        assert (normalise(phys, NORM)["optical_quality"] == 0).all()

    def test_mndwi(self) -> None:
        assert mndwi_from_reflectance(np.array([0.09]), np.array([0.03]))[0] == pytest.approx(0.5)
        assert np.isnan(mndwi_from_reflectance(np.array([0.0]), np.array([0.0]))[0])

    def test_s2_harmonisation_handles_both_baselines(self) -> None:
        """A pre-04.00 product (offset 0) and a post-04.00 one decode to the same rho."""
        old = harmonise_s2_dn(np.array([500.0]), product_offset=0.0, product_quantification=10000.0, norm=NORM)
        new = harmonise_s2_dn(np.array([1500.0]), product_offset=-1000.0, product_quantification=10000.0, norm=NORM)
        assert old[0] == pytest.approx(new[0])
        assert np.isnan(harmonise_s2_dn(np.array([0.0]), product_offset=-1000.0,
                                        product_quantification=10000.0, norm=NORM)[0])

    def test_normalisation_uses_frozen_stats_not_the_chip(self) -> None:
        dark = _layers()
        bright = dict(dark, vv_post_db=dark["vv_post_db"] + 10.0)
        a = normalise(decode_physical(encode_stack(**dark), GRID, NORM), NORM)["sar"][0]
        b = normalise(decode_physical(encode_stack(**bright), GRID, NORM), NORM)["sar"][0]
        ok = a != 0
        assert np.allclose(b[ok] - a[ok], 10.0 / NORM.sar_std_db[0], atol=1e-3)


# ---------------------------------------------------------------------------
# Synthetic events
# ---------------------------------------------------------------------------
class TestSyntheticPhysics:
    @pytest.fixture(scope="class")
    def event(self):
        spec = EventSpec("phys", seed=5, n_basins=2, basin_width_px=256, height_px=320,
                         terrain="hilly", riparian_forest=True, optical="clear", flood_depth_m=(1.5, 2.0),
                         wind_speed_ms=1.0)
        return render_event(spec, GRID, NORM, keep_layers=True)

    def test_render_is_deterministic(self) -> None:
        spec = TINY_EVENTS[0]
        assert np.array_equal(render_event(spec, GRID, NORM).stack, render_event(spec, GRID, NORM).stack)

    def test_open_flood_water_is_dark_and_darkened(self, event) -> None:
        phys = decode_physical(event.stack, GRID, NORM)
        lc, flood = event.layers["landcover"], event.layers["flood_truth"]
        open_flood = flood & (lc != 10) & (phys.validity == 0)
        dry = ~flood & (phys.validity == 0) & (lc != 80)
        assert np.nanmedian(phys.gamma0_db[0][open_flood]) < -18
        assert np.nanmedian(phys.gamma0_db[0][dry]) > -14
        assert np.nanmedian(phys.gamma0_db[4][open_flood]) < -6, "change term must see the flood"

    def test_flooded_forest_is_brighter_than_dry_forest(self, event) -> None:
        """Double bounce (docs/01 section 7): flooded forest is BRIGHTER, not darker."""
        phys = decode_physical(event.stack, GRID, NORM)
        lc, flood = event.layers["landcover"], event.layers["flood_truth"]
        flooded_forest = flood & (lc == 10)
        dry_forest = ~flood & (lc == 10) & (phys.validity == 0)
        if flooded_forest.sum() < 20:
            pytest.skip("no flooded forest in this render")
        assert np.nanmedian(phys.gamma0_db[0][flooded_forest]) > np.nanmedian(phys.gamma0_db[0][dry_forest])

    def test_label_excludes_permanent_water(self, event) -> None:
        assert not (event.layers["label"] & event.layers["permanent"]).any()

    def test_validity_carries_no_steep_slope_flag(self, event) -> None:
        phys = decode_physical(event.stack, GRID, NORM)
        assert not (phys.validity & ValidityFlag.STEEP_SLOPE).any()
        assert (phys.validity[event.layers["permanent"]] & ValidityFlag.PERMANENT_WATER).all()

    def test_turbid_flood_water_has_positive_mndwi(self, event) -> None:
        phys = decode_physical(event.stack, GRID, NORM)
        open_flood = event.layers["flood_truth"] & (event.layers["landcover"] != 10) & (phys.optical_quality > 0.9)
        assert np.nanmedian(phys.mndwi[open_flood]) > 0.3

    def test_basins_tile_the_scene(self, event) -> None:
        total = sum(g.area for g in event.basins.values())
        assert total == pytest.approx(event.spec.width_px * event.spec.height_px * 100.0, rel=1e-6)

    def test_look_azimuth_convention(self) -> None:
        assert look_azimuth_for("descending") == pytest.approx(103.0)
        assert look_azimuth_for("ascending") == pytest.approx(257.0)

    def test_wind_raises_open_water_backscatter(self) -> None:
        calm = EventSpec("calm", seed=9, n_basins=1, basin_width_px=256, height_px=256, flood_depth_m=(2.0,), wind_speed_ms=0.5)
        windy = dataclasses.replace(calm, event_id="windy", wind_speed_ms=7.0)
        def flood_vv(spec):
            ev = render_event(spec, GRID, NORM, keep_layers=True)
            m = ev.layers["flood_truth"] & (ev.layers["landcover"] != 10)
            return np.nanmedian(ev.layers["vv_post_db"][m])
        assert flood_vv(windy) > flood_vv(calm) + 5.0, "Bragg roughening (docs/01 section 6)"


# ---------------------------------------------------------------------------
# Chipping and the manifest
# ---------------------------------------------------------------------------
class TestChipping:
    def test_windows_are_complete(self) -> None:
        windows = enumerate_windows(300, 260, 128, 64)
        assert all(r + 128 <= 300 and c + 128 <= 260 for r, c in windows)
        assert len(windows) == 3 * 3

    def test_stack_round_trips_through_geotiff(self, tiny_benchmark) -> None:
        path = tiny_benchmark.root / "events" / TINY_EVENTS[0].event_id / "stack.tif"
        stack, grid, tags = read_event_stack(path)
        assert stack.shape[0] == N_CHIP_BANDS and grid.epsg == TINY_EVENTS[0].epsg
        assert np.array_equal(stack, render_event(TINY_EVENTS[0], GRID, NORM).stack)
        assert tags["event_id"] == TINY_EVENTS[0].event_id

    def test_records_carry_statistics_and_basins(self, tiny_benchmark) -> None:
        records = tiny_benchmark.records
        assert records and all(r.valid_fraction >= TINY_DATA.min_valid_fraction for r in records)
        assert {r.event_id for r in records} == {s.event_id for s in TINY_EVENTS}
        assert all(r.basin_id.startswith(r.event_id) for r in records)
        eval_chips = [r for r in records if r.eval_grid]
        assert all(r.window[0] % 128 == 0 and r.window[1] % 128 == 0 for r in eval_chips)

    def test_manifest_round_trip_and_hash(self, tiny_benchmark) -> None:
        loaded = load_manifest(tiny_benchmark.manifest_path)
        assert [r.chip_id for r in loaded] == sorted(r.chip_id for r in tiny_benchmark.records)
        assert all(Path(r.raster_path).is_file() for r in loaded)
        base = tiny_benchmark.manifest_path.parent
        assert manifest_sha256(loaded, base) == tiny_benchmark.manifest_sha256
        meta = json.loads(tiny_benchmark.manifest_path.with_name("manifest.meta.json").read_text())
        assert meta["manifest_sha256"] == tiny_benchmark.manifest_sha256

    def test_hash_is_location_independent(self, tiny_benchmark, tmp_path) -> None:
        moved = tmp_path / "moved"
        import shutil
        shutil.copytree(tiny_benchmark.root, moved)
        loaded = load_manifest(moved / "chips" / "manifest.parquet")
        assert manifest_sha256(loaded, moved / "chips") == tiny_benchmark.manifest_sha256

    def test_duplicate_chip_ids_are_rejected(self, tmp_path) -> None:
        rec = ChipRecord("c0", "b0", "e0", tmp_path / "x.tif", (0, 0, 8, 8), 0.0, 1.0, False, 1, "ascending")
        with pytest.raises(ValueError, match="duplicate"):
            write_manifest([rec, rec], tmp_path / "m.parquet")

    def test_missing_columns_are_all_named(self, tmp_path) -> None:
        import pandas as pd
        path = tmp_path / "bad.csv"
        pd.DataFrame({"chip_id": ["a"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"basin_id.*event_id"):
            load_manifest(path)

    def test_extract_chips_respects_min_valid_fraction(self, tiny_benchmark) -> None:
        path = tiny_benchmark.root / "events" / TINY_EVENTS[0].event_id / "stack.tif"
        strict = dataclasses.replace(TINY_DATA, min_valid_fraction=0.999)
        rendered = render_event(TINY_EVENTS[0], GRID, NORM)
        records, counts = extract_chips(path, event_id="x", basins=rendered.basins, data_cfg=strict,
                                        grid_cfg=GRID, relative_orbit=1, orbit_state="descending",
                                        has_optical=True)
        assert counts["dropped_low_valid"] > 0
        assert all(r.valid_fraction >= 0.999 for r in records)


# ---------------------------------------------------------------------------
# Splits on real (tiny) data
# ---------------------------------------------------------------------------
class TestSplitsOnBenchmark:
    def _assign(self, bench, buffer_m=300.0):
        import geopandas as gpd
        recs = bench.records
        basins = gpd.read_file(bench.basins_path)
        report: dict = {}
        a = assign_basins({r.chip_id: r.basin_id for r in recs},
                          {r.chip_id: (r.centroid_x, r.centroid_y) for r in recs},
                          basins, buffer_m=buffer_m, chip_size_m=1280.0, report=report)
        return a, report

    def test_lobo_split_is_disjoint_and_buffered(self, tiny_benchmark) -> None:
        a, report = self._assign(tiny_benchmark)
        assert a.train_basins and a.val_basins and a.test_basins
        for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not set(getattr(a, f"{x}_basins")) & set(getattr(a, f"{y}_basins"))
        assert report["dropped"].get("footprint_crosses_boundary", 0) > 0

    def test_larger_buffer_drops_more(self, tiny_benchmark) -> None:
        _, small = self._assign(tiny_benchmark, 100.0)
        _, large = self._assign(tiny_benchmark, 1500.0)
        assert large["n_kept"] < small["n_kept"]

    def test_paired_row13_keeps_test_chips_and_leaks(self, tiny_benchmark) -> None:
        a, _ = self._assign(tiny_benchmark)
        recs = tiny_benchmark.records
        leaky = random_chip_split([r.chip_id for r in recs], {r.chip_id: r.basin_id for r in recs},
                                  fixed_test_chips=a.test_chips,
                                  match_counts={"train": (len(a.train_chips), 0), "val": (len(a.val_chips), 0)})
        assert leaky.test_chips == a.test_chips
        assert len(leaky.train_chips) == len(a.train_chips)
        assert set(leaky.train_basins) & set(leaky.test_basins), "the leak row 13 measures"
        assert test_pixels_covered_by_training(a, recs) == 0.0, "blocked split shares no pixels"

    def test_frozen_split_binds_the_manifest(self, tiny_benchmark, tmp_path) -> None:
        a, _ = self._assign(tiny_benchmark)
        a = dataclasses.replace(a, manifest_sha256=tiny_benchmark.manifest_sha256)
        path = tmp_path / "splits.json"
        freeze_splits(a, path, metadata={"note": "test"})
        assert load_splits(path, expected_manifest_sha256=tiny_benchmark.manifest_sha256) == a
        with pytest.raises(ValueError, match="manifest"):
            load_splits(path, expected_manifest_sha256="0" * 64)

    def test_norm_stats_come_from_the_given_chips(self, tiny_benchmark) -> None:
        recs = [r for r in tiny_benchmark.records if r.eval_grid]
        stats = compute_norm_stats(recs, NORM, GRID)
        assert -20 < stats.sar_mean_db[0] < -5 and stats.sar_std_db[0] > 0
        assert stats.optical_offset == NORM.optical_offset, "only the SAR statistics are replaced"
