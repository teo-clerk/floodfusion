"""GEOID-Flood: pinned, verified labels; events before the benchmark; classes mapped onto our grid (offline)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from floodfusion.data import geoid
from floodfusion.data.chipping import AnalysisGrid


def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_verifies_every_file_and_extracts_only_label_members(tmp_path: Path, monkeypatch) -> None:
    shard = _tar({"EMSR151-1/label/EMSR151-1-0_label.tif": b"x", "EMSR151-1/s1grd/EMSR151-1-0_s1grd_post.tif": b"y",
                  "../evil/EMSR151-1-0_label.tif": b"z", "EMSR151-1/label/EMSR152-1-0_label.tif": b"w"})
    blobs = {"geoid-flood/tile_catalog.parquet": b"catalog", "shard_index.json.gz": b"index",
             "geoid-flood/shards/train/label/train-label-00000-of-00001.tar": shard,
             "geoid-flood/shards/val/label/val-label-00000-of-00001.tar": _tar({}),
             "geoid-flood/shards/test/label/test-label-00000-of-00001.tar": _tar({})}
    monkeypatch.setattr(geoid, "FILES", {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()})
    calls = []

    def get(url: str) -> bytes:
        calls.append(url)
        return blobs[url.split(f"/resolve/{geoid.REVISION}/", 1)[1]]

    done = geoid.download(tmp_path, get=get)
    assert done["geoid-flood/shards/train/label/train-label-00000-of-00001.tar"] == 1
    assert (tmp_path / "EMSR151-1" / "label" / "EMSR151-1-0_label.tif").read_bytes() == b"x"
    assert not (tmp_path / "EMSR151-1" / "s1grd").exists() and not (tmp_path.parent / "evil").exists()
    assert all(u.startswith(f"https://huggingface.co/datasets/links-ads/geoid-flood/resolve/{geoid.REVISION}/")
               for u in calls)
    assert set(geoid.download(tmp_path, get=lambda u: pytest.fail("resumed: nothing is fetched again")).values()) == {-1}
    monkeypatch.setattr(geoid, "FILES", {"shard_index.json.gz": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        geoid.download(tmp_path / "other", get=get)


def test_events_before_the_benchmark_with_a_usable_tile_minus_kuro_siwo() -> None:
    catalog = pd.DataFrame({"event_id": ["EMSR151", "EMSR151", "EMSR200", "EMSR655", "EMSR656", "EMSR700", "EMSR300"],
                            "is_valid": [True, True, True, True, True, True, False],
                            "invalid_pixel_frac": [0.1, 0.99, 0.2, 0.3, 0.1, 0.1, 0.1]})
    assert geoid.selected_events(catalog, ["EMSR200"]) == ["EMSR151", "EMSR655"]


def test_post_time_is_the_one_most_tiles_carry() -> None:
    index = {"shards": [{"path": "geoid-flood/shards/train/s1grd/x.tar", "members": [
        "EMSR151-1/s1grd/EMSR151-1-0_s1grd_post_20160113T063833.tif",
        "EMSR151-1/s1grd/EMSR151-1-1_s1grd_post_20160113T063833.tif",
        "EMSR151-1/s1grd/EMSR151-1-2_s1grd_post_20160114T063000.tif",
        "EMSR151-1/s1grd/EMSR151-1-2_s1grd_pre_20151201T063833.tif",
        "EMSR151-2/s1grd/EMSR151-2-0_s1grd_post_20160120T000000.tif",
        "EMSR151-2/s1grd/EMSR151-2-1_s1grd_post_20160119T000000.tif"]},
        {"path": "geoid-flood/shards/train/label/y.tar", "members": ["EMSR151-1/label/EMSR151-1-0_label.tif"]}]}
    times = geoid.post_times(index)
    assert times[("EMSR151", 1)] == datetime(2016, 1, 13, 6, 38, 33, tzinfo=UTC)
    assert times[("EMSR151", 2)] == datetime(2016, 1, 19, tzinfo=UTC)          # a tie: the earliest


GRID = AnalysisGrid(epsg=32629, transform=(10.0, 0.0, 645_120.0, 0.0, -10.0, 6_062_080.0), width=40, height=30)


def _label_tile(path: Path, values: np.ndarray, x0: float = 645_120.0, y0: float = 6_062_080.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", count=1, height=values.shape[0],
                       width=values.shape[1], crs=CRS.from_epsg(32629), transform=from_origin(x0, y0, 10.0, 10.0),
                       nodata=255) as dst:
        dst.write(values.astype(np.uint8), 1)


def _aoi(tmp_path: Path) -> geoid.GfAoi:
    a = np.full((30, 40), 255)
    a[:, :20] = 0
    a[:10, :10] = 2                                                     # flood
    a[20:, :5] = 1                                                      # permanent water
    b = np.full((30, 40), 255)
    b[:, 15:] = 2                                                       # overlaps A: A's values must win there
    _label_tile(tmp_path / "A.tif", a)
    _label_tile(tmp_path / "B.tif", b)
    footprint = box(-7.0, 54.0, -6.0, 55.0)
    return geoid.GfAoi("EMSR151", 1, ((str(tmp_path / "A.tif"), footprint), (str(tmp_path / "B.tif"), footprint)),
                       datetime(2016, 1, 13, tzinfo=UTC))


def test_classes_mosaic_first_valid_and_map_to_flood_not_flood_and_unlabelled(tmp_path: Path) -> None:
    aoi = _aoi(tmp_path)
    classes = geoid.label_classes_on_grid(aoi, GRID)
    assert (classes[:10, :10] == 2).all() and (classes[:, 10:20][10:] == 0).all()
    assert (classes[:, 20:] == 2).all() and (classes[15, 15] == 0)       # B fills only A's gaps
    label = geoid._label(classes)
    assert label[0, 0] == 1.0 and label[25, 0] == 0.0 and np.isfinite(label).all()
    assert np.isnan(geoid._label(np.array([255], np.uint8)))[0]
    assert aoi.name == "GF_EMSR151_01"


def test_reference_flags_class_1_as_permanent_water(tmp_path: Path) -> None:
    ref = geoid._reference(_aoi(tmp_path), GRID, [], [])
    assert ref.permanent[25, 0] and not ref.permanent[0, 0]
    assert ref.label[25, 0] == 0.0 and ref.inside_aoi.all()


def test_planning_shares_count_flood_among_mapped_pixels_only(tmp_path: Path) -> None:
    from floodfusion.data.kurosiwo import Tile

    a = np.full((512, 512), 255)
    a[:256] = 0
    a[:64, :64] = 2
    _label_tile(tmp_path / "C.tif", a, x0=600_000.0, y0=6_005_120.0)
    aoi = geoid.GfAoi("EMSR151", 1, ((str(tmp_path / "C.tif"), box(-180, -90, 180, 90)),), datetime(2016, 1, 1))
    flood, mapped = geoid._shares(aoi, Tile("t", (0, 0, 1, 1), 0.0, (32629, 600_000.0, 6_000_000.0)))
    assert mapped == pytest.approx(0.5) and flood == pytest.approx(64 * 64 / (256 * 512), rel=0.05)


def test_kuro_siwo_codes_cover_every_annotated_event(tmp_path: Path) -> None:
    f = tmp_path / "events.yaml"
    f.write_text("events:\n  EMSR118_Spain: {post: [2015-02-02]}\n  EMSR1111002_Cameroon: {post: []}\n")
    assert geoid.kurosiwo_codes(f) == ["EMSR1111002", "EMSR118"]


def test_load_aois_needs_labels_on_disk_and_a_post_time(tmp_path: Path) -> None:
    import shapely

    raw = tmp_path
    (raw / "geoid-flood").mkdir()
    geom = shapely.to_wkb(box(-7, 54, -6.9, 54.1))
    pd.DataFrame({"event_id": ["EMSR151", "EMSR151", "EMSR160"], "tile_id": ["1", "1", "1"],
                  "bbox_id": ["0", "1", "0"], "geometry": [geom, geom, geom], "is_valid": [True, True, True],
                  "invalid_pixel_frac": [0.1, 0.1, 0.1]}).to_parquet(raw / "geoid-flood" / "tile_catalog.parquet")
    _label_tile(raw / "EMSR151-1" / "label" / "EMSR151-1-1_label.tif", np.zeros((4, 4)))
    _label_tile(raw / "EMSR160-1" / "label" / "EMSR160-1-0_label.tif", np.zeros((4, 4)))
    index = {"shards": [{"path": "geoid-flood/shards/train/s1grd/x.tar",
                         "members": ["EMSR151-1/s1grd/EMSR151-1-1_s1grd_post_20160113T063833.tif"]}]}
    (raw / "shard_index.json.gz").write_bytes(gzip.compress(json.dumps(index).encode()))
    aois = geoid.load_aois(raw, ["EMSR151", "EMSR160"])
    assert [(a.event, a.aoi, len(a.tiles)) for a in aois] == [("EMSR151", 1, 1)]   # EMSR160: no post time
