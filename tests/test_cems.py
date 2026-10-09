"""CEMS Rapid Mapping client: API parsing, nearest product, safe package extraction, source times."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from floodfusion.data import cems

PKG = "EMSR999_AOI01_DEL_PRODUCT"


def _activation() -> dict:
    def product(kind, monit, when, sensor):
        part = "PRODUCT" if monit == 0 else f"MONIT{monit:02d}"
        return {"type": kind, "monitoringNumber": monit, "version": 1,
                "downloadPath": f"https://rapidmapping.emergency.copernicus.eu/backend/EMSR999/AOI01/{kind}_{part}/x.zip",
                "images": [{"sensorName": sensor, "acquisitionTime": when}]}
    return {"code": "EMSR999", "aois": [{"name": "Testville", "number": 1, "products": [
        product("DEL", 1, "2023-05-18T16:55:00", "RADARSAT-2"),
        product("GRA", 0, "2023-05-22T10:27:00", "Pleiades"),
        product("DEL", 0, "2023-05-17T16:56:00", "COSMO-SkyMed"),
    ]}]}


class TestApi:
    def test_activation_code_is_validated(self) -> None:
        with pytest.raises(ValueError, match="activation code"):
            cems.fetch_activation("EMSR664&x=1", get=lambda *a, **k: b"{}")

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(LookupError):
            cems.fetch_activation("EMSR123", get=lambda *a, **k: json.dumps({"results": []}).encode())

    def test_products_are_parsed_filtered_and_ordered(self) -> None:
        get = lambda url, **kw: json.dumps({"results": [_activation()]}).encode()  # noqa: E731
        products = cems.list_products(cems.fetch_activation("EMSR999", get=get))
        assert [(p.product_type, p.monitoring) for p in products] == [("DEL", 0), ("DEL", 1)]
        assert products[1].tag == "EMSR999_AOI01_DEL_MONIT01" and products[0].aoi_name == "Testville"
        assert products[0].acquisition_times == (datetime(2023, 5, 17, 16, 56, tzinfo=UTC),)
        assert len(cems.list_products(_activation(), product_type=None)) == 3

    def test_nearest_product_and_its_time_offset(self) -> None:
        products = cems.list_products(_activation())
        product, hours = cems.nearest_product(products, datetime(2023, 5, 18, 16, 58, tzinfo=UTC))
        assert product.monitoring == 1 and hours == pytest.approx(0.05)
        product, hours = cems.nearest_product(products, datetime(2023, 5, 17, 5, 0, tzinfo=UTC))
        assert product.monitoring == 0 and hours == pytest.approx(-11.933, abs=1e-3)


def _source_dbf(tmp: Path) -> bytes:
    import pandas as pd
    import pyogrio

    frame = pd.DataFrame({"source_nam": ["Sentinel-2", "COSMO-SkyMed", "COSMO-SkyMed"],
                          "src_date": ["06/08/2022", "17/05/2023", "17/05/2023"],
                          "source_tm": ["T10:06:00Z", "T16:56:00Z", "T17:14:00Z"],
                          "eventphase": ["Pre-event", "Post-event", "Post-event"], "src_id": [1, 2, 3]})
    pyogrio.write_dataframe(frame, tmp / "source.dbf", driver="ESRI Shapefile")
    return (tmp / "source.dbf").read_bytes()


def _package(tmp: Path) -> bytes:
    flood = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"event_type": "5-Flood", "dmg_src_id": 2},
         "geometry": {"type": "Polygon", "coordinates": [[[12.0, 44.3], [12.01, 44.3], [12.01, 44.31], [12.0, 44.3]]]}},
        {"type": "Feature", "properties": {"event_type": "5-Flood", "dmg_src_id": 3},
         "geometry": {"type": "Polygon", "coordinates": [[[12.1, 44.3], [12.11, 44.3], [12.11, 44.31], [12.1, 44.3]]]}},
    ]}
    aoi = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
           "geometry": {"type": "Polygon", "coordinates": [[[11.9, 44.2], [12.2, 44.2], [12.2, 44.4], [11.9, 44.2]]]}}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{PKG}_observedEventA_v1.json", json.dumps(flood))
        z.writestr(f"{PKG}_areaOfInterestA_v1.json", json.dumps(aoi))
        z.writestr(f"{PKG}_source_v1.dbf", _source_dbf(tmp))
        z.writestr(f"{PKG}_observedEventA_v1.kmz", b"not kept")
        z.writestr("Ready-to-print maps/map.pdf", b"not kept")
        z.writestr(f"nested/{PKG}_observedEventA_v1.json", b"nested paths are never written")
        z.writestr(f"{PKG}_v1/{PKG}_imageFootprintA_v1.json", b"{}")       # the one allowed folder level
        z.writestr(f"{PKG}_v1/deeper/{PKG}_hydrographyL_v1.json", b"too deep")
        z.writestr(f"../{PKG}_hydrographyA_v1.json", b"traversal")
    return buf.getvalue()


@pytest.fixture()
def product_dir(tmp_path):
    product = cems.list_products(_activation())[0]
    package = _package(tmp_path)
    calls = []
    out = cems.download_product(product, tmp_path / "cems", get=lambda url, **kw: calls.append(url) or package)
    return out, product, calls, package


class TestPackages:
    def test_only_listed_members_are_extracted(self, product_dir, tmp_path) -> None:
        out, _, calls, _ = product_dir
        assert out == tmp_path / "cems" / "EMSR999" / "AOI01" / "EMSR999_AOI01_DEL_PRODUCT" and len(calls) == 1
        assert sorted(p.name for p in out.iterdir()) == sorted([
            f"{PKG}_observedEventA_v1.json", f"{PKG}_areaOfInterestA_v1.json", f"{PKG}_source_v1.dbf",
            f"{PKG}_imageFootprintA_v1.json", "manifest.json"])
        assert not (tmp_path / "cems" / "EMSR999" / "AOI01" / f"{PKG}_hydrographyA_v1.json").exists()

    def test_download_is_resumable(self, product_dir, tmp_path) -> None:
        out, product, _, _ = product_dir
        again = cems.download_product(product, tmp_path / "cems", get=lambda *a, **k: pytest.fail("refetched"))
        assert again == out

    def test_flood_polygons_carry_their_source_time(self, product_dir) -> None:
        out, *_ = product_dir
        assert cems.source_times(out) == {2: datetime(2023, 5, 17, 16, 56, tzinfo=UTC),
                                          3: datetime(2023, 5, 17, 17, 14, tzinfo=UTC)}
        flood = cems.observed_flood(out)
        assert len(flood) == 2 and flood.crs.to_epsg() == 4326
        assert sorted(t.minute for t in flood["source_time"]) == [14, 56]
        assert cems.aoi_geometry(out).bounds == pytest.approx((11.9, 44.2, 12.2, 44.4))


class _Response:
    def __init__(self, status: int, body: bytes = b"", location: str = "") -> None:
        self.status_code, self.body, self.headers = status, body, {"Location": location}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.body


class TestHttp:
    @pytest.mark.parametrize("url", ["http://rapidmapping.emergency.copernicus.eu/x",
                                     "https://rapidmapping.emergency.copernicus.eu.evil.example/x",
                                     "https://evil.example/backend/x"])
    def test_requests_leave_only_for_the_api_host(self, url) -> None:
        with pytest.raises(ValueError, match="refusing"):
            cems._http_get(url, cap=10)

    def _patch(self, monkeypatch, responses: dict[str, _Response]) -> list[str]:
        import requests

        seen = []

        def fake_get(url, **kw):
            seen.append(url)
            assert kw["allow_redirects"] is False
            return responses[url.split("?", 1)[0]]

        monkeypatch.setattr(requests, "get", fake_get)
        return seen

    def test_one_redirect_into_the_bucket_is_followed(self, monkeypatch) -> None:
        api = "https://rapidmapping.emergency.copernicus.eu/backend/x.zip"
        s3 = "https://rapidmapping.s3.amazonaws.com/x.zip"
        seen = self._patch(monkeypatch, {api: _Response(302, location=s3 + "?sig=1"), s3: _Response(200, b"zip")})
        assert cems._http_get(api, cap=100) == b"zip" and seen == [api, s3 + "?sig=1"]

    def test_a_redirect_anywhere_else_is_refused(self, monkeypatch) -> None:
        api = "https://rapidmapping.emergency.copernicus.eu/backend/x.zip"
        self._patch(monkeypatch, {api: _Response(302, location="https://evil.example/x.zip")})
        with pytest.raises(ValueError, match="refusing"):
            cems._http_get(api, cap=100)

    def test_bodies_are_capped(self, monkeypatch) -> None:
        api = "https://rapidmapping.emergency.copernicus.eu/backend/x.json"
        self._patch(monkeypatch, {api: _Response(200, b"x" * 101)})
        with pytest.raises(OSError, match="larger than"):
            cems._http_get(api, cap=100)

    def test_transient_errors_are_retried_with_backoff(self, monkeypatch) -> None:
        import requests

        api = "https://rapidmapping.emergency.copernicus.eu/backend/x.json"
        answers = iter([_Response(502), _Response(503), _Response(200, b"ok")])
        monkeypatch.setattr(requests, "get", lambda url, **kw: next(answers))
        waits = []
        assert cems._http_get(api, cap=100, sleep=waits.append) == b"ok" and waits == [2.0, 4.0]

    def test_permanent_errors_and_exhausted_retries_raise(self, monkeypatch) -> None:
        import requests

        api = "https://rapidmapping.emergency.copernicus.eu/backend/x.json"
        monkeypatch.setattr(requests, "get", lambda url, **kw: _Response(404))
        with pytest.raises(OSError, match="404"):
            cems._http_get(api, cap=100, sleep=lambda s: None)
        monkeypatch.setattr(requests, "get", lambda url, **kw: _Response(502))
        with pytest.raises(OSError, match="502"):
            cems._http_get(api, cap=100, sleep=lambda s: None)
