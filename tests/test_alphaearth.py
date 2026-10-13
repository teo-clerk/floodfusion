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
