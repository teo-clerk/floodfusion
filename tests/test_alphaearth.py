"""AlphaEarth context (v1.2, deferred): year rule, de-quantisation, PCA storage, orientation score (offline)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from floodfusion.data import alphaearth as ae


def test_the_year_before_the_event_never_overlaps_it() -> None:
    assert ae.aef_year(datetime(2023, 5, 20, tzinfo=UTC)) == 2022
    assert ae.aef_year(datetime(2023, 2, 10, tzinfo=UTC)) == 2021        # 60 days back lands in 2022
    assert ae.aef_year(datetime(2027, 6, 1, tzinfo=UTC)) == ae.LAST_YEAR
    assert ae.aef_year(datetime(2017, 6, 1, tzinfo=UTC)) is None         # 2016: before the archive


def test_dequantisation_and_nodata() -> None:
    raw = np.zeros((64, 1, 2), np.int8)
    raw[0, 0, 0], raw[1, 0, 0] = 127, -64
    raw[5, 0, 1] = ae.NODATA
    emb = ae.dequantise(raw)
    assert emb[0, 0, 0] == pytest.approx((127 / 127.5) ** 2) and emb[1, 0, 0] == pytest.approx(-((64 / 127.5) ** 2))
    assert np.isnan(emb[:, 0, 1]).all()                                  # one band missing: the whole pixel


def test_https_urls_only_for_the_mirror() -> None:
    url = ae.https_url("s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2024/32N/x-0-0.tiff")
    assert url == "https://data.source.coop/tge-labs/aef/v1/annual/2024/32N/x-0-0.tiff"
    with pytest.raises(ValueError):
        ae.https_url("s3://other-bucket/x.tiff")
    with pytest.raises(ValueError):
        ae.https_url("s3://us-west-2.opendata.source.coop/other/x.tiff")


def test_files_are_chosen_by_year_and_footprint() -> None:
    index = pd.DataFrame({"path": ["a", "b", "c"], "year": [2022, 2022, 2021], "wgs84_west": [10, 20, 10],
                          "wgs84_east": [11, 21, 11], "wgs84_south": [44, 44, 44], "wgs84_north": [45, 45, 45]})
    assert ae.files_for(index, 2022, (10.5, 44.2, 10.7, 44.4)) == ["a"]


def test_pca_storage_round_trips_and_marks_missing_pixels() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 64))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pca = ae.fit_pca(x, n=8)
    assert np.allclose(np.asarray(pca["components"]) @ np.asarray(pca["components"]).T, np.eye(8), atol=1e-8)
    emb = x[:6].T.reshape(64, 2, 3).copy()
    emb[:, 1, 2] = np.nan
    stored = ae.project(emb, pca)
    assert stored.dtype == np.int8 and stored.shape == (8, 2, 3) and (stored[:, 1, 2] == ae.NODATA).all()
    expect = (x[0] - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T
    np.testing.assert_allclose(stored[:, 0, 0] / ae.AEF_SCALE, expect, atol=1 / ae.AEF_SCALE)


def test_orientation_score_drops_when_the_embedding_is_upside_down() -> None:
    rng = np.random.default_rng(1)
    water = rng.random((40, 40)) < 0.15                                  # irregular water: no mirror image of itself
    emb = rng.normal(0, 0.1, size=(64, 40, 40))
    emb[0][water] += 1.0
    scores = ae.orientation_check(emb, water)
    assert scores["auc_as_read"] > 0.99 and scores["auc_flipped"] < 0.6


def _bottom_up_tiff(path, n_rows: int = 32) -> None:
    """A 64-band int8 TIFF stored bottom-up (positive y pixel size), band 1 = its file row, with a x2 overview."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    data = np.zeros((64, n_rows, n_rows), dtype=np.int8)
    data[0] = np.arange(n_rows, dtype=np.int8)[:, None]                 # file row 0 is the SOUTH edge
    with rasterio.open(path, "w", driver="GTiff", dtype="int8", count=64, height=n_rows, width=n_rows,
                       crs=CRS.from_epsg(32632), transform=Affine(10, 0, 500_000, 0, 10, 5_000_000),
                       nodata=ae.NODATA) as dst:
        dst.write(data)
        dst.build_overviews([2], Resampling.nearest)


def test_a_bottom_up_tiff_is_read_north_up_from_its_20_m_overview(tmp_path, monkeypatch) -> None:
    from floodfusion.data.chipping import AnalysisGrid

    _bottom_up_tiff(tmp_path / "aef.tiff")
    monkeypatch.setattr(ae, "_VSI", "")
    grid = AnalysisGrid(epsg=32632, transform=(20, 0, 500_000, 0, -20, 5_000_320), width=16, height=16)
    band = ae._read_band(str(tmp_path / "aef.tiff"), 1, grid)
    assert band.shape == (16, 16)
    assert band[0].min() >= 29 and band[-1].max() <= 2                 # north at the top, as on the stack grid
    assert np.all(np.diff(band[:, 0].astype(int)) <= 0)


def _index(paths_years) -> pd.DataFrame:
    return pd.DataFrame([{"path": p, "year": y, "wgs84_west": 8.9, "wgs84_south": 45.0, "wgs84_east": 9.1,
                          "wgs84_north": 45.2} for p, y in paths_years])


def _stack(root, name: str, post: str = "2024-05-20T05:00:00Z"):
    import json

    d = root / "events" / name
    d.mkdir(parents=True)
    (d / "event.json").write_text(json.dumps({"epsg": 32632, "transform": [10, 0, 500_000, 0, -10, 5_000_080],
                                              "width": 8, "height": 8, "s1_post_datetime": post}))
    return d


_PATH = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/32N/x.tiff"


def test_files_mosaic_first_valid_and_each_stack_records_its_year_and_files(tmp_path, monkeypatch) -> None:
    def fake_band(url, band, grid):
        layer = np.full((grid.height, grid.width), 64 if url.endswith("x.tiff") else -64, np.int8)
        if url.endswith("x.tiff"):
            layer[:, :4] = ae.NODATA                                     # the first file misses the west half
        return layer

    monkeypatch.setattr(ae, "_read_band", fake_band)
    checked = []
    monkeypatch.setattr(ae, "_preflight", lambda url: checked.append(url) or url)     # offline: no HEAD request
    second = _PATH.replace("x.tiff", "y.tiff")
    d = _stack(tmp_path, "S1")
    emb, record = ae.stack_embeddings(d, _index([(_PATH, 2023), (second, 2023), (_PATH, 2022)]))
    assert record == {"status": "ok", "year": 2023, "files": [_PATH, second]}
    assert len(checked) == 2 and all(u.startswith("https://data.source.coop/") for u in checked)   # once per file
    assert emb.shape == (64, 8, 8) and np.isfinite(emb).all()
    assert emb[0, 0, 7] == pytest.approx((64 / 127.5) ** 2) and emb[0, 0, 0] == pytest.approx(-(64 / 127.5) ** 2)
    assert ae.stack_embeddings(d, _index([(_PATH, 2021)]))[1]["status"] == "no AlphaEarth file of 2023"
    early = _stack(tmp_path, "S0", post="2017-03-01T00:00:00Z")
    assert ae.stack_embeddings(early, _index([]))[1]["status"] == "no AlphaEarth year before the event"


def test_the_sidecar_holds_the_projected_components_and_a_provenance_record(tmp_path, monkeypatch) -> None:
    import json

    import rasterio

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(64, 8, 8)).astype(np.float32)
    emb[:, 0, 0] = np.nan
    monkeypatch.setattr(ae, "stack_embeddings", lambda d, index: (emb, {"status": "ok", "year": 2023, "files": []}))
    pca = ae.fit_pca(emb.reshape(64, -1).T[1:])
    d = _stack(tmp_path, "S1")
    assert ae.build_aef(d, None, pca) == "ok"
    with rasterio.open(d / "aef.tif") as src:
        stored = src.read()
        assert src.count == ae.N_COMPONENTS and src.nodata == ae.NODATA
    assert np.array_equal(stored, ae.project(emb, pca)) and stored[0, 0, 0] == ae.NODATA
    record = json.loads((d / "aef.json").read_text())
    assert record["pca_sha256"] == ae.pca_digest(pca) and record["valid_fraction"] == pytest.approx(63 / 64)
    assert ae.build_aef(d, None, pca) == "done-before"

    monkeypatch.setattr(ae, "stack_embeddings", lambda d, index: (None, {"status": "no AlphaEarth file of 2023"}))
    assert ae.build_aef(d, None, pca, overwrite=True) == "no AlphaEarth file of 2023"
    assert not (d / "aef.tif").exists()                                  # a stale layer is removed


def test_the_cli_fits_the_pca_builds_the_sidecars_and_checks_orientation(tmp_path, monkeypatch) -> None:
    import json

    rng = np.random.default_rng(1)
    index = tmp_path / "index.parquet"
    _index([(_PATH, 2023)]).to_parquet(index)
    for n in range(3):
        _stack(tmp_path, f"S{n}")
    water = np.zeros((8, 8), bool)
    water[:, :4] = True

    def fake_embeddings(d, idx):
        e = rng.normal(size=(64, 8, 8)).astype(np.float32)
        e[0, water] += 4.0
        return e, {"status": "ok", "year": 2023, "files": [_PATH]}

    monkeypatch.setattr(ae, "stack_embeddings", fake_embeddings)
    monkeypatch.setattr(ae, "_water", lambda d: water)
    pca_path, common = tmp_path / "pca.json", ["--root", str(tmp_path), "--index", str(index)]
    assert ae.main(["fit", *common, "--pca", str(pca_path), "--n-stacks", "2", "--per-stack", "50"]) == 0
    pca = json.loads(pca_path.read_text())
    assert len(pca["stacks"]) == 2 and pca["n_samples"] == 100 and pca["scale"] == ae.AEF_SCALE
    assert ae.main(["build", *common, "--pca", str(pca_path), "--workers", "1"]) == 0
    assert all((tmp_path / "events" / f"S{n}" / "aef.tif").is_file() for n in range(3))
    assert ae.main(["check", *common, "--n", "2"]) == 0

    monkeypatch.setattr(ae, "build_aef", lambda d, index, pca: (_ for _ in ()).throw(OSError("timeout")))
    assert ae.main(["build", *common, "--pca", str(pca_path)]) == 1
