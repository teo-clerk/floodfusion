"""Kuro Siwo: pinned download, AOIs and dates, tiles, groups, shifts, labels, and the water relabelling."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from floodfusion.data import kurosiwo as ks
from floodfusion.data.chipping import AnalysisGrid, write_event_stack
from floodfusion.data.encoding import N_CHIP_BANDS, ChipBand
from floodfusion.data.records import ChipRecord, load_manifest, write_manifest

ROOT = "KuroSiwo-annotations-" + ks.ANNOTATIONS_COMMIT
GRID = AnalysisGrid(epsg=32633, transform=(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_900_000.0), width=8, height=8)


def _grid_box_lonlat(grid: AnalysisGrid):
    from pyproj import Transformer

    to = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    x0, y0, x1, y1 = grid.bounds
    lon, lat = to.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(lon), min(lat), max(lon), max(lat)


def _write_layer(path: Path, geoms) -> None:
    import geopandas as gpd

    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": list(range(len(geoms)))}, geometry=list(geoms), crs=4326).to_file(path)


def _raw_aoi(raw: Path, event: str = "EMSR1_Test", aoi: str = "01", version: str = "r1_v1", *,
             flood=None, hydro=None, region=None) -> Path:
    base = raw / "polygons" / event / aoi / version
    west, south, east, north = _grid_box_lonlat(GRID)
    region = region if region is not None else box(west, south, east, north)
    _write_layer(base / "aoi" / "aoi.shp", [region])
    _write_layer(base / "event" / "event.shp", [flood if flood is not None else box(west, south, (west + east) / 2, north)])
    _write_layer(base / "hydro" / "hydroA.shp", [hydro if hydro is not None else box((west + east) / 2, south, east, north)])
    return base


class TestDownload:
    @staticmethod
    def _tarball(members: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def test_only_shapefile_parts_of_the_polygon_tree_are_extracted(self, tmp_path, monkeypatch) -> None:
        data = self._tarball({
            f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/event.shp": b"shp",
            f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/event.dbf": b"dbf",
            f"{ROOT}/README.md": b"readme",
            f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/../../../../../evil.shp": b"no",
            f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/run.sh": b"no",
        })
        monkeypatch.setattr(ks, "ANNOTATIONS_SHA256", hashlib.sha256(data).hexdigest())
        assert ks.download(tmp_path, get=lambda url: data) == 2
        files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
        assert files == ["polygons/.complete", "polygons/EMSR1_Test/01/r1_v1/event/event.dbf",
                         "polygons/EMSR1_Test/01/r1_v1/event/event.shp"]
        assert ks.download(tmp_path, get=lambda url: pytest.fail("refetched")) == 0   # resumable

    def test_links_devices_and_foreign_roots_are_never_written(self, tmp_path, monkeypatch) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            good = tarfile.TarInfo(f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/event.shp")
            good.size = 3
            tar.addfile(good, io.BytesIO(b"shp"))
            link = tarfile.TarInfo(f"{ROOT}/polygons/EMSR1_Test/aoi/01/r1_v1/event/event.dbf")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(link)
            other = tarfile.TarInfo("Other-repo-abc/polygons/EMSR1_Test/aoi/01/r1_v1/event/event.prj")
            other.size = 3
            tar.addfile(other, io.BytesIO(b"prj"))
        data = buf.getvalue()
        monkeypatch.setattr(ks, "ANNOTATIONS_SHA256", hashlib.sha256(data).hexdigest())
        assert ks.download(tmp_path, get=lambda url: data) == 1
        assert not any(p.is_symlink() for p in tmp_path.rglob("*"))

    def test_a_tarball_matching_nothing_is_an_error(self, tmp_path, monkeypatch) -> None:
        data = self._tarball({f"{ROOT}/README.md": b"readme"})
        monkeypatch.setattr(ks, "ANNOTATIONS_SHA256", hashlib.sha256(data).hexdigest())
        with pytest.raises(OSError, match="no shapefile member"):
            ks.download(tmp_path, get=lambda url: data)

    def test_a_tarball_with_another_hash_is_refused_before_writing(self, tmp_path) -> None:
        with pytest.raises(OSError, match="sha256"):
            ks.download(tmp_path, get=lambda url: b"not the pinned tarball")
        assert not any(tmp_path.iterdir())


class TestEventsAndGroups:
    def test_the_latest_annotation_version_is_used(self, tmp_path) -> None:
        _raw_aoi(tmp_path, version="r1_v1")
        _raw_aoi(tmp_path, version="r2_v1")
        _raw_aoi(tmp_path, aoi="02", version="r1_v3")
        aois = ks.list_aois(tmp_path)
        assert [(a.aoi, a.layers.name) for a in aois] == [("01", "r2_v1"), ("02", "r1_v3")]
        assert aois[0].stack_prefix == "KS_EMSR1_Test_01"

    def test_dates_are_read_and_undated_events_left_out(self, tmp_path) -> None:
        path = tmp_path / "events.yaml"
        path.write_text("events:\n  EMSR1_Test: {post: [2021-02-03]}\n  EMSR2_Two: {post: [2022-03-07, 2022-03-02]}\n"
                        "  EMSR3_Undated: {post: []}\n", encoding="utf-8")
        assert ks.load_event_dates(path) == {"EMSR1_Test": (date(2021, 2, 3),),
                                             "EMSR2_Two": (date(2022, 3, 2), date(2022, 3, 7))}

    def test_events_within_the_distance_share_a_group_by_single_linkage(self) -> None:
        geoms = {"B": box(10.0, 45.0, 10.1, 45.1), "A": box(10.9, 45.0, 11.0, 45.1),   # ~63 km apart
                 "C": box(11.8, 45.0, 11.9, 45.1),                                     # ~63 km from A
                 "Z": box(30.0, 0.0, 30.1, 0.1)}
        groups = ks.event_groups(geoms, distance_km=100.0)
        assert groups["A"] == groups["B"] == groups["C"] == "KS_A"
        assert groups["Z"] == "KS_Z"


class TestTiles:
    REGION = box(12.0, 45.0, 12.2, 45.1)    # ~15.7 x 11.1 km

    def test_tiles_lie_inside_the_region_and_carry_their_flood_fraction(self) -> None:
        flood = box(12.0, 45.0, 12.2, 45.02)
        tiles = ks.candidate_tiles(self.REGION, flood, aoi="KS_X_01")
        # >= 90 % of each UTM square lies in the region (its lon/lat bounding box is slightly larger).
        assert tiles and all(box(*t.bbox).intersection(self.REGION).area / box(*t.bbox).area >= 0.8 for t in tiles)
        assert len({t.tile_id for t in tiles}) == len(tiles)
        assert any(t.flood_fraction > 0 for t in tiles) and any(t.flood_fraction == 0 for t in tiles)
        assert all(0.0 <= t.flood_fraction <= 1.0 for t in tiles)

    def test_sampling_caps_wet_and_dry_tiles_and_is_reproducible(self) -> None:
        tiles = [ks.Tile("KS_X_01", (0, 0, 1, 1), 0.5 if i % 2 else 0.0, (32633, float(i), 0.0)) for i in range(40)]
        first = ks.sample_tiles(tiles, "EMSR1_Test")
        assert first == ks.sample_tiles(list(reversed(tiles)), "EMSR1_Test")
        assert sum(t.flood_fraction >= ks.FLOOD_TILE_FRACTION for t in first) == ks.N_FLOOD_TILES
        assert sum(t.flood_fraction < ks.FLOOD_TILE_FRACTION for t in first) == ks.N_DRY_TILES
        assert ks.sample_tiles(tiles, "EMSR2_Other") != first


def test_offsets_are_measured_on_the_training_tiles_plus_a_seeded_draw() -> None:
    cands = [ks.Tile("KS_X_01", (0, 0, 1, 1), 0.0, (32633, float(i), 0.0)) for i in range(100)]
    train = cands[:5]
    sample = ks.shift_sample(train, cands, "KS_X_01", n=20)
    assert len(sample) == 20 and set(train) <= set(sample)
    assert sample == ks.shift_sample(train, list(reversed(cands)), "KS_X_01", n=20)
    assert ks.shift_sample(train, cands[:8], "KS_X_01", n=20) == tuple(sorted(cands[:8], key=lambda t: t.tile_id))


class TestShifts:
    @staticmethod
    def _estimate(shift, n_grids=20, ci=0.4):
        return {"shift_px": shift, "n_grids": n_grids, "ci_halfwidth_px": ci, "median_shift": shift}

    def test_label_and_optical_move_on_their_own_measured_offsets(self) -> None:
        out = ks.choose_shift({"label": self._estimate([1, 0]), "optical": self._estimate([0, 2]),
                               "label_vs_optical": self._estimate([1, -2])})
        assert out["label_shift_px"] == [1, 0] and out["optical_shift_px"] == [0, 2]

    def test_an_imprecise_or_thin_estimate_moves_nothing(self) -> None:
        out = ks.choose_shift({"label": self._estimate([2, 0], ci=1.6), "optical": self._estimate([0, 2], n_grids=5)})
        assert out["label_shift_px"] == [0, 0] and out["label_source"] == "none"
        assert out["optical_shift_px"] == [0, 0] and out["optical_source"] == "none"


def test_label_is_flood_inside_the_aoi_and_nan_outside(tmp_path) -> None:
    import geopandas as gpd

    west, south, east, north = _grid_box_lonlat(GRID)
    aoi = gpd.GeoDataFrame(geometry=[box(west, south, east, (south + north) / 2)], crs=4326)       # southern half
    flood = gpd.GeoDataFrame(geometry=[box(west, south, (west + east) / 2, north)], crs=4326)       # western half
    label = ks.label_on_grid(aoi, flood, GRID)
    assert np.isnan(label[:4]).all()                       # northern half outside the AOI
    assert (label[4:, :4] == 1).all() and (label[4:, 4:] == 0).all()


def test_relabel_water_adds_hydro_on_the_same_windows(tmp_path) -> None:
    from floodfusion.config import ValidityFlag, load_config

    raw = tmp_path / "raw"
    _raw_aoi(raw)
    src = tmp_path / "flood" / "events" / "KS_EMSR1_Test_01_0_0"
    src.mkdir(parents=True)
    stack = np.zeros((N_CHIP_BANDS, GRID.height, GRID.width), dtype=np.int16)
    stack[ChipBand.VV_POST] = -1500
    stack[ChipBand.LABEL, :, :4] = 1                       # flood: western half
    stack[ChipBand.LABEL, 0, 7] = -32768                   # unlabelled stays unlabelled
    stack[ChipBand.VALIDITY, 1, 7] = int(ValidityFlag.PERMANENT_WATER)
    write_event_stack(src / "stack.tif", stack, GRID)
    (src / "basins.gpkg").write_bytes(b"")
    record = ChipRecord("KS_EMSR1_Test_01_0_0_r00000_c00000", "KS_EMSR1_Test", "KS_EMSR1_Test_01_0_0",
                        src / "stack.tif", (0, 0, 8, 8), 0.5, 1.0, False, 95, "descending")
    write_manifest([record], src / "chips.parquet")
    (src / "event.json").write_text(json.dumps({"kurosiwo": {"event": "EMSR1_Test", "aoi": "01"}}), encoding="utf-8")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")

    assert ks.relabel_water(tmp_path / "flood", tmp_path / "water", raw, cfg) == 1

    from floodfusion.data.chipping import read_event_stack

    out, _, _ = read_event_stack(tmp_path / "water" / "events" / src.name / "stack.tif")
    assert (out[ChipBand.LABEL, :, :4] == 1).all()                     # flood kept
    assert out[ChipBand.LABEL, 2, 6] == 1                              # hydro added
    assert out[ChipBand.LABEL, 0, 7] == -32768 and out[ChipBand.LABEL, 1, 7] == 0   # unlabelled / permanent kept
    np.testing.assert_array_equal(np.delete(out, ChipBand.LABEL, axis=0), np.delete(stack, ChipBand.LABEL, axis=0))
    [water] = load_manifest(tmp_path / "water" / "events" / src.name / "chips.parquet")
    assert water.chip_id == record.chip_id and water.window == record.window
    assert water.positive_fraction > record.positive_fraction
