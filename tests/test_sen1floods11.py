"""Sen1Floods11 route: metadata, splits, the verified downloader and the label reference layer.

The end-to-end re-ingest against a local archive is in ``test_ingest_offline.py``
(``TestSen1Floods11Route``), next to the archive fixture it shares with the event route.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from floodfusion.data.chipping import build_analysis_grid
from floodfusion.data.sen1floods11 import (
    _check_url,
    chip_reference,
    download,
    list_hand_chips,
    official_split,
    parse_metadata,
    parse_split_csv,
)

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
LIST = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"


def _feature(location: str, s1: str, s2: str, orbit: int, state: str) -> dict:
    return {"type": "Feature", "geometry": None, "properties": {
        "location": location, "s1_date": s1, "s2_date": s2, "rel_orbit_num": float(orbit), "orbit": state}}


def write_metadata(path: Path, features: list[dict]) -> Path:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return path


class TestMetadata:
    def test_events_keyed_by_chip_prefix(self, tmp_path) -> None:
        meta = write_metadata(tmp_path / "m.geojson", [
            _feature("Bolivia", "2018/02/15", "2018/02/15", 156, "DESCENDING"),
            _feature("Bolivia", "2018/02/15", "2018/02/15", 156, "DESCENDING"),   # duplicated row
            _feature("Cambodia", "2018/08/05", "2018/08/04", 26, "ASCENDING"),
        ])
        events = parse_metadata(meta)
        assert set(events) == {"Bolivia", "Mekong"}  # chip files call Cambodia "Mekong"
        mekong = events["Mekong"]
        assert (mekong.s1_date, mekong.s2_date, mekong.relative_orbit, mekong.orbit_state) == (
            date(2018, 8, 5), date(2018, 8, 4), 26, "ascending")
        assert mekong.group == "S1F11_Mekong"

    def test_conflicting_rows_are_refused(self, tmp_path) -> None:
        meta = write_metadata(tmp_path / "m.geojson", [
            _feature("Spain", "2019/09/17", "2019/09/18", 110, "DESCENDING"),
            _feature("Spain", "2019/09/17", "2019/09/18", 111, "DESCENDING"),
        ])
        with pytest.raises(ValueError, match="conflicting"):
            parse_metadata(meta)


class TestSplits:
    def test_split_csv_gives_chip_ids(self) -> None:
        text = "Bolivia_103757_S1Hand.tif,Bolivia_103757_LabelHand.tif\nSri-Lanka_249079_S1Hand.tif,x\n\n"
        assert parse_split_csv(text) == ("Bolivia_103757", "Sri-Lanka_249079")

    def _write_splits(self, root: Path, rows: dict[str, list[str]]) -> None:
        d = root / "splits" / "flood_handlabeled"
        d.mkdir(parents=True)
        names = {"train": "flood_train_data", "val": "flood_valid_data", "test": "flood_test_data",
                 "bolivia": "flood_bolivia_data"}
        for split, name in names.items():
            (d / f"{name}.csv").write_text("".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\n" for c in rows.get(split, [])))

    def test_official_split_is_within_event(self, tmp_path) -> None:
        self._write_splits(tmp_path, {"train": ["Ghana_1", "Ghana_2"], "test": ["Ghana_3"], "bolivia": ["Bolivia_9"]})
        split = official_split(tmp_path)
        assert split == {"Ghana_1": "train", "Ghana_2": "train", "Ghana_3": "test", "Bolivia_9": "bolivia"}

    def test_a_chip_in_two_splits_is_refused(self, tmp_path) -> None:
        self._write_splits(tmp_path, {"train": ["Ghana_1"], "test": ["Ghana_1"]})
        with pytest.raises(ValueError, match="both"):
            official_split(tmp_path)


def _md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class FakeBucket:
    """The two Google Cloud Storage endpoints the downloader uses, served from a dict."""

    def __init__(self, objects: dict[str, bytes], *, corrupt: str | None = None) -> None:
        self.objects, self.corrupt, self.fetched = objects, corrupt, []

    def __call__(self, url: str, params: dict | None = None) -> bytes:
        _check_url(url)
        if url == LIST:
            prefix = params["prefix"]
            items = [{"name": n, "size": str(len(b)), "md5Hash": _md5(b)}
                     for n, b in sorted(self.objects.items()) if n.startswith(prefix)]
            return json.dumps({"items": items}).encode()
        name = url.split("/sen1floods11/", 1)[1]
        self.fetched.append(name)
        data = self.objects[name]
        return data + b"x" if name == self.corrupt else data


def _bucket_objects() -> dict[str, bytes]:
    objects = {"v1.1/Sen1Floods11_Metadata.geojson": b"{}"}
    for name in ("flood_train_data", "flood_valid_data", "flood_test_data", "flood_bolivia_data"):
        objects[f"v1.1/splits/flood_handlabeled/{name}.csv"] = b"a,b\n"
    objects["v1.1/splits/flood_handlabeled/other.csv"] = b"not wanted"
    for chip in ("Ghana_1", "Spain_2"):
        objects[f"v1.1/data/flood_events/HandLabeled/LabelHand/{chip}_LabelHand.tif"] = chip.encode()
    return objects


class TestDownload:
    def test_fetches_verifies_and_resumes(self, tmp_path) -> None:
        bucket = FakeBucket(_bucket_objects())
        assert download(tmp_path, get=bucket) == {"fetched": 7, "skipped": 0}
        assert (tmp_path / "data/flood_events/HandLabeled/LabelHand/Spain_2_LabelHand.tif").read_bytes() == b"Spain_2"
        assert not (tmp_path / "splits/flood_handlabeled/other.csv").exists()
        assert [c.chip_id for c in list_hand_chips(tmp_path)] == ["Ghana_1", "Spain_2"]
        again = FakeBucket(_bucket_objects())
        assert download(tmp_path, get=again) == {"fetched": 0, "skipped": 7} and not again.fetched

    def test_md5_mismatch_is_fatal_and_nothing_is_written(self, tmp_path) -> None:
        name = "v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_1_LabelHand.tif"
        with pytest.raises(OSError, match="md5"):
            download(tmp_path, get=FakeBucket(_bucket_objects(), corrupt=name))
        assert not (tmp_path / name.removeprefix("v1.1/")).exists()

    def test_object_names_cannot_escape_the_root(self, tmp_path) -> None:
        objects = _bucket_objects()
        objects["v1.1/data/flood_events/HandLabeled/LabelHand/../../../../../evil.tif"] = b"x"
        with pytest.raises(ValueError, match="escapes"):
            download(tmp_path / "raw", get=FakeBucket(objects))
        assert not (tmp_path / "evil.tif").exists()

    @pytest.mark.parametrize("url", ["http://storage.googleapis.com/sen1floods11/x",
                                     "https://storage.googleapis.com/other-bucket/x",
                                     "https://storage.googleapis.com.evil.example/sen1floods11/x",
                                     "https://evil.example/sen1floods11/x"])
    def test_urls_outside_the_bucket_are_refused(self, url) -> None:
        with pytest.raises(ValueError, match="refusing"):
            _check_url(url)

    def test_layer_names_are_validated(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="layer"):
            download(tmp_path, layers=("../x",), get=FakeBucket(_bucket_objects()))


class TestChipReference:
    """A 4326 label chip on a UTM grid: water minus permanent water; -1 and off-chip are NaN."""

    BOUNDS = (10.200, 44.400, 10.209, 44.409)   # ~100 x 100 px at the dataset's 8.98e-5 deg

    def _label(self, tmp_path: Path) -> Path:
        n = 100
        label = np.zeros((n, n), np.int16)
        label[:, :50] = 1           # water in the west
        label[:10, :] = -1          # an unlabelled northern strip
        path = tmp_path / "Testland_1_LabelHand.tif"
        with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="int16", crs="EPSG:4326",
                           transform=from_bounds(*self.BOUNDS, n, n)) as dst:
            dst.write(label, 1)
        return path

    def _jrc(self, tmp_path: Path) -> list[dict]:
        occ = np.zeros((90, 90), np.uint8)
        occ[:, :9] = 95             # a permanent channel along the west edge
        path = tmp_path / "occ.tif"
        with rasterio.open(path, "w", driver="GTiff", width=90, height=90, count=1, dtype="uint8", crs="EPSG:4326",
                           transform=from_bounds(*self.BOUNDS, 90, 90), nodata=255) as dst:
            dst.write(occ, 1)
        return [{"id": "jrc", "bbox": list(self.BOUNDS), "assets": {"occurrence": {"href": str(path)}}}]

    def test_label_semantics_on_the_grid(self, tmp_path) -> None:
        grid = build_analysis_grid((10.195, 44.395, 10.214, 44.414), 10.0)   # larger than the chip
        ref = chip_reference(self._label(tmp_path), grid, self._jrc(tmp_path), [])
        assert ref.label.shape == ref.inside_aoi.shape == (grid.height, grid.width)
        assert not ref.inside_aoi.all() and ref.inside_aoi.mean() > 0.2      # off-chip pixels exist
        assert np.isnan(ref.label[~ref.inside_aoi]).all()
        labelled = ref.inside_aoi & np.isfinite(ref.label)
        assert set(np.unique(ref.label[labelled])) == {0.0, 1.0}
        assert np.isnan(ref.label[ref.inside_aoi]).mean() == pytest.approx(0.1, abs=0.03)  # the -1 strip
        assert ref.permanent.any() and not (ref.label[ref.permanent] == 1).any()           # flood != permanent
        assert (ref.landcover == 0).all()                                                  # no WorldCover given


_OPTICAL_ESTIMATE = {"n_windows": 600, "n_grids": 20, "median_shift": [0.1, 0.3], "shift_px": [0, 0],
                     "ci_halfwidth_px": 0.4}


class TestEventShiftDecision:
    """One whole-pixel shift per event for label + optical (DATA_CARD: label registration)."""

    OPTICAL = _OPTICAL_ESTIMATE

    def test_a_precise_label_estimate_decides(self) -> None:
        from floodfusion.data.sen1floods11 import choose_event_shift

        label = {"n_windows": 900, "n_grids": 30, "median_shift": [0.2, 2.6], "shift_px": [0, 3], "ci_halfwidth_px": 0.4}
        out = choose_event_shift(self.OPTICAL, label)
        assert out["shift_px"] == [0, 3] and out["source"] == "label"
        assert out["optical"] == self.OPTICAL and out["label"] == label

    def test_an_imprecise_or_missing_label_estimate_falls_back_to_the_optical_one(self) -> None:
        from floodfusion.data.sen1floods11 import choose_event_shift

        vague = {"n_windows": 90, "n_grids": 6, "median_shift": [-0.9, 1.0], "shift_px": [-1, 1], "ci_halfwidth_px": 4.2}
        assert choose_event_shift(self.OPTICAL, vague)["source"] == "optical"
        assert choose_event_shift(self.OPTICAL, vague)["shift_px"] == [0, 0]
        empty = {"n_windows": 0, "n_grids": 0, "median_shift": [float("nan")] * 2, "shift_px": [0, 0],
                 "ci_halfwidth_px": float("nan")}
        assert choose_event_shift(self.OPTICAL, empty)["source"] == "optical"

    def test_too_few_chips_never_decide(self) -> None:
        from floodfusion.data.sen1floods11 import choose_event_shift

        two_chips = {"n_windows": 400, "n_grids": 2, "median_shift": [0.0, 2.0], "shift_px": [0, 2], "ci_halfwidth_px": 0.1}
        assert choose_event_shift(self.OPTICAL, two_chips)["source"] == "optical"
