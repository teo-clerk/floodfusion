"""GFM-style product package: layer semantics, Cloud-Optimized GeoTIFFs, the STAC Item."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodfusion.predict import PRODUCT_FLAGS
from floodfusion.products import GFM_LAYERS, gfm_layers, write_package


def _bit(*names: str) -> int:
    return sum(1 << PRODUCT_FLAGS[n] for n in names)


@pytest.fixture()
def scene():
    """A 600 x 600 grid: flood in the west, a permanent channel, shadow and a forest patch."""
    n = 600
    prob = np.full((n, n), 0.05, np.float32)
    prob[:, :250] = 0.9
    flags = np.zeros((n, n), np.uint16)
    flags[:, 300:310] |= _bit("PERMANENT_WATER")
    flags[:40, :] |= _bit("SHADOW")
    flags[500:, 500:] |= _bit("DENSE_FOREST")
    flags[100:110, :] |= _bit("LOW_CONFIDENCE")
    prob[590:, :] = np.nan
    flags[590:, :] |= _bit("NODATA")
    return prob, flags


class TestLayers:
    def test_semantics(self, scene) -> None:
        prob, flags = scene
        layers = gfm_layers(prob, flags, tau=0.25)
        assert tuple(layers) == GFM_LAYERS
        flood, water = layers["observed_flood"].data, layers["observed_water"].data
        assert flood[200, 100] == 1 and flood[200, 400] == 0 and flood[10, 100] == 255   # shadow: no information
        assert flood[200, 305] == 0 and water[200, 305] == 1                              # permanent water is not flood
        assert layers["reference_water"].data[200, 305] == 1
        excl = layers["exclusion_mask"].data
        assert excl[10, 400] == 1 and excl[550, 550] == 1 and excl[595, 0] == 1 and excl[200, 400] == 0
        assert layers["likelihood"].data[200, 100] == 90 and layers["likelihood"].data[595, 0] == 255
        assert layers["advisory_flags"].data[105, 400] & 1 and not layers["advisory_flags"].data[200, 400] & 1
        # Forest is excluded, but its evidence is kept: the likelihood is still there.
        assert layers["likelihood"].data[550, 550] == 5
        assert np.array_equal(layers["flags"].data, flags)


class TestPackage:
    def test_cogs_and_stac_item(self, scene, tmp_path) -> None:
        prob, flags = scene
        profile = {"width": 600, "height": 600, "crs": "EPSG:32632", "transform": from_origin(720000, 4920000, 10, 10)}
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"weights")
        meta = {"event_id": "EMSR664_tile", "s1_post_datetime": "2023-05-18T16:58:50+00:00",
                "s1_post_frames": ["S1A_post_rtc"], "s1_pre_frames": ["S1A_pre_rtc"], "s2_items": [],
                "relative_orbit": 44, "orbit_state": "ascending"}
        item_path = write_package(tmp_path / "pkg", prob, flags, tau=0.25, profile=profile, meta=meta, weights=weights,
                                  provenance={"git_sha": "abc", "config_sha256": "cfg", "version": "1.0.0"})
        item = json.loads(item_path.read_text())
        assert set(item["assets"]) == set(GFM_LAYERS)
        for name, asset in item["assets"].items():
            path = item_path.parent / asset["href"]
            assert asset["href"] == f"./{name}.tif" and "cloud-optimized" in asset["type"]
            with rasterio.open(path) as src:
                assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG" and src.overviews(1)
                assert src.crs.to_epsg() == 32632 and src.dtypes[0] == asset["raster:bands"][0]["data_type"]
        props = item["properties"]
        assert props["datetime"].startswith("2023-05-18T16:58") and props["proj:code"] == "EPSG:32632"
        assert props["sat:relative_orbit"] == 44 and props["floodfusion:tau_star"] == 0.25
        assert props["floodfusion:model_sha256"] == hashlib.sha256(b"weights").hexdigest()
        assert "nodata" not in item["assets"]["exclusion_mask"]["raster:bands"][0]   # absent, never null
        assert {link["href"].rsplit("/", 1)[-1] for link in item["links"] if link["rel"] == "derived_from"} == {
            "S1A_post_rtc", "S1A_pre_rtc"}
        w, s, e, n = item["bbox"]
        assert 11.0 < w < e < 12.5 and 44.0 < s < n < 44.6


@pytest.mark.network
def test_item_validates_against_the_published_schemas(scene, tmp_path) -> None:
    import pystac

    prob, flags = scene
    profile = {"width": 600, "height": 600, "crs": "EPSG:32632", "transform": from_origin(720000, 4920000, 10, 10)}
    item_path = write_package(tmp_path / "pkg", prob, flags, tau=0.25, profile=profile,
                              meta={"event_id": "x", "s1_post_datetime": "2023-05-18T16:58:50+00:00"},
                              provenance={"git_sha": "abc", "version": "1.0.0"})
    pystac.Item.from_file(str(item_path)).validate()
