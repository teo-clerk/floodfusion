"""Real-data ingest, run end to end offline against local GeoTIFF/XML "archive" assets.

Every asset an ingest reads from Planetary Computer is mimicked by a small local file with
the same conventions: RTC gamma0 in linear power (nodata -32768) on a UTM grid, the GRD
item whose ``schema-product-vv`` points at the RFI annotation (the real product annotation
fixture sits at ``annotation/iw-vv.xml``), Sentinel-2 L2A DN with the baseline-04.00
offset, GLO-30 / JRC / WorldCover in geographic coordinates, CEMS-like label and AOI
vectors, and HydroBASINS-like polygons.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_bounds
from shapely.geometry import box

from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.chipping import read_event_stack
from floodfusion.data.encoding import ChipBand, decode_physical
from floodfusion.data.ingest import (
    EventConfig,
    IngestSources,
    build_dataset,
    grd_id_for_rtc,
    ingest_event,
    load_event_config,
    product_annotation_href,
)
from floodfusion.data.records import load_manifest
from floodfusion.data.stac import InMemoryCatalogue
from floodfusion.config import ValidityFlag

ANNOTATION = Path(__file__).parent / "fixtures" / "s1a_iw_grd_vv_20231107t170700_annotation_trimmed.xml"
BBOX = (10.20, 44.40, 10.24, 44.43)          # ~3.2 x 3.3 km inside the fixture's frame
WIDE = (10.15, 44.35, 10.30, 44.48)           # extent of the geographic "archive" rasters
GRD_ID = "S1A_IW_GRDH_1SDV_20231107T170700_20231107T170725_051114_062A12"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _write(path: Path, data: np.ndarray, bounds, crs: str, nodata=None) -> str:
    h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype=data.dtype,
                       crs=crs, transform=from_bounds(*bounds, w, h), nodata=nodata) as dst:
        dst.write(data, 1)
    return str(path)


def _utm_bounds():
    from pyproj import Transformer

    x0, y0, x1, y1 = Transformer.from_crs(4326, 32632, always_xy=True).transform_bounds(*WIDE)
    return (x0, y0, x1, y1)


def _item(item_id, collection, when, assets, bbox=WIDE, **props):
    return {"id": item_id, "collection": collection, "bbox": list(bbox),
            "properties": {"datetime": when, **props},
            "assets": {k: {"href": v} for k, v in assets.items()}}


def _s1_props():
    return {"sat:relative_orbit": 117, "sat:orbit_state": "ascending", "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"]}


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """Local files + an in-memory catalogue that together look like Planetary Computer."""
    root = tmp_path_factory.mktemp("archive")
    rng = np.random.default_rng(0)
    utm = _utm_bounds()
    n = 400
    # Flood in the western half: dark post-event gamma0 there; land ~ -8 dB VV.
    _, xx = np.mgrid[0:n, 0:n]
    flooded = xx < n // 2
    items = []
    for when, date_str, is_post in (("post", "2023-11-07T17:07:13Z", True), ("pre", "2023-10-14T17:07:13Z", False)):
        assets = {}
        for pol, land_db in (("vv", -8.0), ("vh", -15.0)):
            db = np.full((n, n), land_db) + rng.normal(0, 1.0, (n, n))
            if is_post:
                db = np.where(flooded, land_db - 12.0, db)
            linear = (10 ** (db / 10)).astype(np.float32)
            linear[:3, :] = -32768.0  # a nodata stripe
            assets[pol] = _write(root / f"rtc_{when}_{pol}.tif", linear, utm, "EPSG:32632", nodata=-32768.0)
        items.append(_item(f"S1_{when}_rtc" if not is_post else f"{GRD_ID}_rtc", "sentinel-1-rtc", date_str,
                           assets, **_s1_props()))
    # GRD item: schema-product-vv -> RFI file; product annotation one directory up (PC layout).
    safe = root / "safe" / "annotation"
    (safe / "rfi").mkdir(parents=True)
    shutil.copy(ANNOTATION, safe / "iw-vv.xml")
    (safe / "rfi" / "rfi-iw-vv.xml").write_text("<rfi/>")
    items.append(_item(GRD_ID, "sentinel-1-grd", "2023-11-07T17:07:13Z",
                       {"schema-product-vv": str(safe / "rfi" / "rfi-iw-vv.xml")}, **_s1_props()))
    # Sentinel-2 L2A, baseline 05.09 (offset -1000): turbid water rho (0.09, 0.07, 0.03), land (0.07, 0.30, 0.21).
    s2 = {}
    for band, water, land in (("B03", 0.09, 0.07), ("B08", 0.07, 0.30), ("B11", 0.03, 0.21)):
        rho = np.where(flooded, water, land)
        s2[band] = _write(root / f"s2_{band}.tif", (rho * 10000 + 1000).astype(np.uint16), utm, "EPSG:32632", nodata=0)
    scl = np.where(flooded, 6, 4).astype(np.uint8)
    scl[-60:, -60:] = 9  # a cloud in one corner
    s2["SCL"] = _write(root / "s2_scl.tif", scl, utm, "EPSG:32632", nodata=0)
    items.append(_item("S2B_MSIL2A_20231106T102139_R065_T32TPR_20231106T152052", "sentinel-2-l2a",
                       "2023-11-06T10:21:39Z", s2, **{"eo:cloud_cover": 8.0, "s2:processing_baseline": "05.09"}))
    # Geographic layers: DEM (gentle eastward rise), JRC occurrence (a permanent channel), WorldCover.
    m = 300
    dem = (20.0 + np.tile(np.linspace(0, 30, m), (m, 1))).astype(np.float32)
    items.append(_item("Copernicus_DSM_COG_10_N44_00_E010_00_DEM", "cop-dem-glo-30", "2021-04-22T00:00:00Z",
                       {"data": _write(root / "dem.tif", dem, WIDE, "EPSG:4326")}))
    occ = np.zeros((m, m), np.uint8)
    occ[:, 95:100] = 95
    items.append(_item("10E_50Nv1_3_2020", "jrc-gsw", "2020-07-01T00:00:00Z",
                       {"occurrence": _write(root / "occ.tif", occ, WIDE, "EPSG:4326", nodata=255)}))
    wc = np.full((m, m), 40, np.uint8)
    items.append(_item("ESA_WorldCover_10m_2021_v200_N44E009", "esa-worldcover", "2021-01-01T00:00:00Z",
                       {"map": _write(root / "wc.tif", wc, WIDE, "EPSG:4326", nodata=0)},
                       start_datetime="2021-01-01T00:00:00Z"))
    # Vectors: label = western half of the AOI; AOI = whole bbox minus a northern strip; two basins.
    mid = 0.5 * (BBOX[0] + BBOX[2])
    gpd.GeoDataFrame({"event_type": ["5-Flood"]}, geometry=[box(BBOX[0] - 0.01, BBOX[1] - 0.01, mid, BBOX[3] + 0.01)],
                     crs="EPSG:4326").to_file(root / "label.geojson")
    gpd.GeoDataFrame(geometry=[box(BBOX[0] - 0.01, BBOX[1] - 0.01, BBOX[2] + 0.01, BBOX[3] - 0.004)],
                     crs="EPSG:4326").to_file(root / "aoi.geojson")
    gpd.GeoDataFrame({"HYBAS_ID": [2100001, 2100002]},
                     geometry=[box(10.1, 44.3, mid, 44.5), box(mid, 44.3, 10.4, 44.5)],
                     crs="EPSG:4326").to_file(root / "basins.geojson")
    return root, InMemoryCatalogue.from_dicts(items)


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    base = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return load_config(base, overrides=["data.chip_size=64", "data.train_stride=32", "data.eval_stride=64"])


def _event(root: Path, **kw) -> EventConfig:
    values = dict(event_id="T_po", bbox=BBOX, flood_start=date(2023, 11, 4), flood_end=date(2023, 11, 7),
                  label_path=root / "label.geojson", aoi_path=root / "aoi.geojson",
                  basins_path=root / "basins.geojson", catalogue_url=PC)
    values.update(kw)
    return EventConfig(**values)


@pytest.fixture(scope="module")
def ingested(archive, cfg, tmp_path_factory):
    root, catalogue = archive
    out = tmp_path_factory.mktemp("ingest")
    meta = ingest_event(_event(root), out, cfg, sources=IngestSources(client=catalogue))
    return out, meta


class TestHelpers:
    def test_grd_id(self) -> None:
        assert grd_id_for_rtc(f"{GRD_ID}_rtc") == GRD_ID and grd_id_for_rtc(GRD_ID) == GRD_ID

    def test_product_annotation_href_skips_the_rfi_file(self) -> None:
        item = {"id": "x", "assets": {"schema-product-vv": {
            "href": "https://acc.blob.core.windows.net/s1-grd/X_C9A0/annotation/rfi/rfi-iw-vv.xml?st=1&sig=2"}}}
        assert product_annotation_href(item) == "https://acc.blob.core.windows.net/s1-grd/X_C9A0/annotation/iw-vv.xml?st=1&sig=2"

    @pytest.mark.parametrize("href", ["http://x.blob.core.windows.net/a.tif", "https://evil.example/a.tif",
                                      "/vsicurl/https://evil.example/a.tif", "<VRTDataset/>",
                                      "https://x.blob.core.windows.net.evil.example/a.tif"])
    def test_untrusted_hrefs_are_refused(self, href) -> None:
        from floodfusion.data.sources import check_href

        with pytest.raises(ValueError):
            check_href(href)

    def test_trusted_hrefs_pass(self, tmp_path) -> None:
        from floodfusion.data.sources import check_href

        for href in ("https://sentinel1euwest.blob.core.windows.net/s1-grd/a.xml", str(tmp_path / "a.tif")):
            assert check_href(href) == href

    def test_event_yaml_example_parses(self) -> None:
        example = Path(__file__).resolve().parents[1] / "configs" / "events" / "example.yaml"
        event = load_event_config(example)
        assert event.event_id == "EMSR664_lugo" and event.flood_end == date(2023, 5, 22)
        assert event.label_path.is_absolute()

    def test_event_yaml_rejects_unknown_keys_and_bad_bbox(self, tmp_path) -> None:
        bad = tmp_path / "e.yaml"
        bad.write_text("event_id: x\nbbox: [1, 2, 3, 4]\nflood_start: 2023-01-01\nflood_end: 2023-01-02\n"
                       "label_path: l.gpkg\ncolour: red\n")
        with pytest.raises(ValueError, match="colour"):
            load_event_config(bad)
        bad.write_text("event_id: x\nbbox: [3, 2, 1, 4]\nflood_start: 2023-01-01\nflood_end: 2023-01-02\nlabel_path: l\n")
        with pytest.raises(ValueError, match="bbox"):
            load_event_config(bad)

    def test_non_pc_catalogue_is_refused_for_pixels(self, archive, cfg, tmp_path) -> None:
        root, catalogue = archive
        with pytest.raises(ValueError, match="Planetary Computer"):
            ingest_event(_event(root, catalogue_url="https://catalogue.dataspace.copernicus.eu/stac"), tmp_path, cfg,
                         sources=IngestSources(client=catalogue))

    def test_missing_same_orbit_pre_scene_drops_the_event(self, archive, cfg, tmp_path) -> None:
        root, catalogue = archive
        with pytest.raises(LookupError):
            ingest_event(_event(root, pre_event_window_days=5), tmp_path, cfg, sources=IngestSources(client=catalogue))


class TestIngestedEvent:
    def test_geometry_comes_from_the_annotation(self, ingested) -> None:
        _, meta = ingested
        assert meta["look_azimuth_deg"] == pytest.approx(255.92, abs=0.01)
        lo, hi = meta["incidence_deg"]
        assert 30.0 < lo < hi < 46.5

    def test_stack_layers(self, ingested, cfg) -> None:
        out, meta = ingested
        stack, grid, tags = read_event_stack(out / "events" / "T_po" / "stack.tif")
        assert grid.epsg == 32632 and stack.shape[0] == 16 and tags["look_azimuth_deg"].startswith("255.9")
        chip = decode_physical(stack, cfg.grid, cfg.norm)
        valid = stack[ChipBand.VALIDITY] == 0
        west, east = slice(0, grid.width // 2 - 10), slice(grid.width // 2 + 10, None)
        # Dark flood water in the post image; delta VV strongly negative there only.
        vv_post = stack[ChipBand.VV_POST] / 100.0
        assert np.median(vv_post[:, west][valid[:, west]]) < -17 < np.median(vv_post[:, east][valid[:, east]])
        # S2 offset applied: B11 reflectance ~0.03 on water, ~0.21 on land.
        b11 = chip.reflectance[2]
        assert np.nanmedian(b11[:, west]) == pytest.approx(0.03, abs=0.01)
        assert np.nanmedian(b11[:, east]) == pytest.approx(0.21, abs=0.01)
        assert meta["has_optical"] and meta["s2_offsets"]["boa_add_offset_by_item"]

    def test_label_permanent_water_and_aoi(self, ingested) -> None:
        out, meta = ingested
        stack, _, _ = read_event_stack(out / "events" / "T_po" / "stack.tif")
        validity, label = stack[ChipBand.VALIDITY], stack[ChipBand.LABEL]
        assert (validity & ValidityFlag.PERMANENT_WATER).any()
        assert not ((label == 1) & ((validity & ValidityFlag.PERMANENT_WATER) > 0)).any()
        outside = (validity & ValidityFlag.OUTSIDE_AOI) > 0
        assert outside[:5].all() and not outside[-5:].any()  # northern strip is outside the AOI
        assert 0.2 < meta["flood_fraction_of_valid"] < 0.7

    def test_chips_basins_and_split_crs(self, ingested) -> None:
        out, _ = ingested
        records = load_manifest(out / "events" / "T_po" / "chips.parquet")
        assert records and {r.basin_id for r in records} <= {"HB2100001", "HB2100002"}
        assert all(r.relative_orbit == 117 and r.orbit_state == "ascending" for r in records)
        # Centroids live in the dataset split CRS (EPSG:3035), not the event UTM zone.
        assert all(4.0e6 < r.centroid_x < 4.6e6 and 2.2e6 < r.centroid_y < 2.6e6 for r in records)

    def test_qa_report(self, ingested) -> None:
        out, _ = ingested
        on_disk = json.loads((out / "events" / "T_po" / "event.json").read_text())
        assert set(on_disk["rtc_qa"]) == {"vv_post_db", "vh_post_db", "vv_pre_db", "vh_pre_db"}
        assert {"pre_vs_post", "terrain_simulation_vs_post", "sar_vs_optical_nir", "p1_m1_pass"} <= set(
            on_disk["coregistration_qa"])
        assert (out / "events" / "T_po" / "scenes.json").is_file()

    def test_build_dataset_merges_events(self, ingested, cfg) -> None:
        out, _ = ingested
        digest = build_dataset(out, cfg)
        manifest = out / "chips" / cfg.data.manifest_name
        assert len(digest) == 64 and manifest.is_file()
        basins = gpd.read_file(out / "basins.gpkg")
        assert basins.crs.to_epsg() == 3035 and set(basins["basin_id"]) == {"HB2100001", "HB2100002"}
        assert {r.chip_id for r in load_manifest(manifest)} == {
            r.chip_id for r in load_manifest(out / "events" / "T_po" / "chips.parquet")}


# ---------------------------------------------------------------------------
# The Sen1Floods11 route: same archive, labels from a hand-labelled chip
# ---------------------------------------------------------------------------
S1F11_CHIP = (10.215, 44.405, 10.235, 44.420)   # straddles the archive's flood edge at ~10.225 E


def _s1f11_raw(root: Path, orbit: int = 117) -> Path:
    raw = root / f"raw_{orbit}"
    labels = raw / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    labels.mkdir(parents=True)
    (raw / "Sen1Floods11_Metadata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": None, "properties": {
            "location": "Testland", "s1_date": "2023/11/07", "s2_date": "2023/11/06",
            "rel_orbit_num": float(orbit), "orbit": "ASCENDING"}}]}))
    w, h = 223, 167                          # ~8.98e-5 deg pixels, like the dataset
    lon = np.linspace(S1F11_CHIP[0], S1F11_CHIP[2], w, endpoint=False)
    label = np.where(lon[None, :] < 10.225, 1, 0).repeat(h, axis=0).astype(np.int16)
    label[:12, :] = -1                       # an unlabelled strip
    _write(labels / "Testland_1_LabelHand.tif", label, S1F11_CHIP, "EPSG:4326")
    return raw


@pytest.fixture(scope="module")
def s1f11_ingested(archive, cfg, tmp_path_factory):
    from floodfusion.data.sen1floods11 import ingest_sen1floods11

    _, catalogue = archive
    work = tmp_path_factory.mktemp("s1f11")
    raw, out = _s1f11_raw(work), work / "out"
    report = ingest_sen1floods11(raw, out, cfg, sources=IngestSources(client=catalogue), margin_px=32, workers=2)
    return raw, out, report


class TestSen1Floods11Route:
    def test_report_and_resume(self, s1f11_ingested, archive, cfg) -> None:
        from floodfusion.data.sen1floods11 import ingest_sen1floods11

        raw, out, report = s1f11_ingested
        assert report["summary"]["ok"] == 1 and report["events"]["Testland"]["Testland_1"].startswith("ok")
        again = ingest_sen1floods11(raw, out, cfg, sources=IngestSources(client=archive[1]), margin_px=32)
        assert again["summary"] == {"ok": 0, "done-before": 1, "dropped": 0, "failed": 0}

    def test_labels_inputs_and_group(self, s1f11_ingested, cfg) -> None:
        _, out, _ = s1f11_ingested
        event_dir = out / "events" / "S1F11_Testland_1"
        meta = json.loads((event_dir / "event.json").read_text())
        assert meta["sen1floods11"]["chip_id"] == "Testland_1" and meta["terrain_margin_px"] == 32
        assert meta["has_optical"] and meta["relative_orbit"] == 117
        stack, grid, _ = read_event_stack(event_dir / "stack.tif")
        validity, label = stack[ChipBand.VALIDITY], stack[ChipBand.LABEL]
        outside = (validity & ValidityFlag.OUTSIDE_AOI) > 0
        assert outside.any() and not outside.all()            # the 4326 chip rotated into UTM
        scored = (validity == 0) & (label >= 0)
        west, east = slice(0, grid.width // 3), slice(2 * grid.width // 3, None)
        assert (label[:, west][scored[:, west]] == 1).mean() > 0.95
        assert (label[:, east][scored[:, east]] == 0).mean() > 0.95
        # The post image is dark where the label says water (inputs re-fetched, not copied).
        vv_post = stack[ChipBand.VV_POST] / 100.0
        assert np.median(vv_post[:, west][scored[:, west]]) < -17 < np.median(vv_post[:, east][scored[:, east]])
        records = load_manifest(event_dir / "chips.parquet")
        assert records and {r.basin_id for r in records} == {"S1F11_Testland"}

    def test_dataset_manifest_in_web_mercator(self, s1f11_ingested, cfg) -> None:
        _, out, _ = s1f11_ingested
        assert len(build_dataset(out, cfg, split_epsg=3857)) == 64
        assert (out / "basins.gpkg").is_file()

    def test_no_pass_on_the_named_orbit_drops_the_event(self, archive, cfg, tmp_path) -> None:
        from floodfusion.data.sen1floods11 import ingest_sen1floods11

        report = ingest_sen1floods11(_s1f11_raw(tmp_path, orbit=15), tmp_path / "out", cfg,
                                     sources=IngestSources(client=archive[1]), margin_px=32)
        assert report["summary"]["dropped"] == 1 and "relative orbit 15" in report["events"]["Testland"]["Testland_1"]


def test_missing_grd_annotation_falls_back_to_the_orbit_model(archive) -> None:
    """Planetary Computer lists some RTC frames whose source GRD it does not carry."""
    from floodfusion.data.assembly import read_acquisition_geometry
    from floodfusion.data.stac import search_matched_pairs

    from shapely.geometry import MultiPoint, mapping

    from floodfusion.data.assembly import incidence_on_grid
    from floodfusion.data.chipping import build_analysis_grid
    from floodfusion.data.s1_annotation import parse_annotation

    _, catalogue = archive
    truth = parse_annotation(ANNOTATION.read_bytes())
    swath = MultiPoint(np.column_stack([truth.geolocation["longitude"], truth.geolocation["latitude"]])).convex_hull
    items = [i if i["collection"] != "sentinel-1-rtc" else {**i, "geometry": mapping(swath)}
             for i in catalogue.items if i["collection"] != "sentinel-1-grd"]
    no_grd = InMemoryCatalogue.from_dicts(items)
    event = EventConfig(event_id="T_po", bbox=BBOX, flood_start=date(2023, 11, 4), flood_end=date(2023, 11, 7),
                        label_path=Path("unused"), catalogue_url=PC)
    matched = search_matched_pairs(event.query(), PC, client=no_grd)
    (geometry,) = read_acquisition_geometry(IngestSources(client=no_grd), matched)
    assert geometry.source.startswith("orbit model") and geometry.orbit_state == "ascending"
    assert geometry.look_azimuth_deg == pytest.approx(truth.look_azimuth_deg, abs=0.5)
    grid = build_analysis_grid(BBOX, 10.0)
    modelled, measured = incidence_on_grid([geometry], grid), incidence_on_grid([truth], grid)
    assert np.abs(modelled - measured).max() < 0.8
    with pytest.raises(LookupError):
        read_acquisition_geometry(IngestSources(client=no_grd), matched, fallback=False)


def test_a_corrupt_label_fails_its_chip_and_the_rest_of_the_event_runs(archive, cfg, tmp_path) -> None:
    """Review finding: an event-level step outside the error handling aborted the whole dataset."""
    from floodfusion.data.sen1floods11 import ingest_sen1floods11

    raw = _s1f11_raw(tmp_path)
    labels = raw / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    w, h = 112, 84
    second = (10.205, 44.410, 10.215, 44.4175)
    label = np.zeros((h, w), np.int16)
    label[:, : w // 2] = 1
    _write(labels / "Testland_2_LabelHand.tif", label, second, "EPSG:4326")
    (labels / "Testland_3_LabelHand.tif").write_bytes(b"<VRTDataset>not a GeoTIFF</VRTDataset>")
    report = ingest_sen1floods11(raw, tmp_path / "out", cfg, sources=IngestSources(client=archive[1]),
                                 margin_px=32, workers=3)
    outcomes = report["events"]["Testland"]
    assert outcomes["Testland_3"].startswith("failed: unreadable label")
    assert outcomes["Testland_1"].startswith("ok") and outcomes["Testland_2"].startswith(("ok", "dropped"))
    assert (tmp_path / "out" / "ingest_report.json").is_file() and report["summary"]["failed"] == 1


def test_a_rerun_never_keeps_stale_artefacts(archive, cfg, tmp_path) -> None:
    root, catalogue = archive
    out = tmp_path / "ingest"
    ingest_event(_event(root), out, cfg, sources=IngestSources(client=catalogue))
    event_dir = out / "events" / "T_po"
    (event_dir / "chips.parquet").write_bytes(b"stale")
    ingest_event(_event(root), out, cfg, sources=IngestSources(client=catalogue))
    assert load_manifest(event_dir / "chips.parquet")          # rewritten, readable
