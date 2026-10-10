"""Remote-input guards: each case is a bypass a security review demonstrated, pinned."""

from __future__ import annotations

import pytest

from floodfusion.data.http import TransientHTTPError, capped_get, check_https_url, check_local_path, redact
from floodfusion.data.sources import check_href

PC_BLOB = "https://sentinel1euwest.blob.core.windows.net/s1-grd/a.xml"


class TestUrls:
    @pytest.mark.parametrize("url", [
        "https://evil.example\\@rapidmapping.emergency.copernicus.eu/p.zip",   # urlparse vs urllib3 differential
        "https://user@rapidmapping.emergency.copernicus.eu/p.zip",
        "https://rapidmapping.emergency.copernicus.eu:8443/p.zip",
        "https://rapidmapping.emergency.copernicus.eu/p.zip\n",
        "https://rapidmapping.emergency.copernicus.eu.evil.example/p.zip",
        "http://rapidmapping.emergency.copernicus.eu/p.zip",
    ])
    def test_hostile_urls_are_refused(self, url) -> None:
        with pytest.raises(ValueError, match="refusing"):
            check_https_url(url, hosts=("rapidmapping.emergency.copernicus.eu",))

    def test_plain_urls_pass_with_their_path_prefix(self) -> None:
        url = "https://storage.googleapis.com/sen1floods11/v1.1/x.tif"
        assert check_https_url(url, hosts=("storage.googleapis.com",), path_prefixes=("/sen1floods11/",)) == url
        with pytest.raises(ValueError, match="paths"):
            check_https_url(url, hosts=("storage.googleapis.com",), path_prefixes=("/other/",))
        assert check_https_url(PC_BLOB, suffixes=(".blob.core.windows.net",)) == PC_BLOB


class TestGdalPaths:
    @pytest.mark.parametrize("href", [
        "file:/vsicurl/https://x/y.tif", "GTIFF_DIR:1:/vsicurl/https://x/y.tif", "/VSICURL/https://x/y.tif",
        "file:///vsizip/a.zip/b.tif", "/data/../vsimem/x", "<VRTDataset/>", "NETCDF:/tmp/x.nc:var",
        "https://evil.example\\@sentinel1euwest.blob.core.windows.net/a.tif",
    ])
    def test_virtual_and_prefixed_paths_are_refused(self, href) -> None:
        with pytest.raises(ValueError, match="refusing"):
            check_href(href)

    def test_plain_paths_and_file_urls_pass(self, tmp_path) -> None:
        for href in (str(tmp_path / "a.tif"), "relative/b.tif", f"file://{tmp_path}/c.tif", PC_BLOB):
            assert check_href(href) == href
        assert check_local_path(f"file://{tmp_path}/c.tif") == f"{tmp_path}/c.tif"


def test_redaction_covers_urllib3_error_text() -> None:
    text = ("HTTPSConnectionPool(host='x.blob.core.windows.net', port=443): Max retries exceeded with "
            "url: /s1/a.tif?st=2023&sig=SECRET (Caused by ...) and https://h/p?sig=SECRET2")
    out = redact(text)
    assert "SECRET" not in out and out.count("<redacted>") == 2


class _Response:
    def __init__(self, status: int, body: bytes = b"", location: str = "") -> None:
        self.status_code, self.body, self.headers = status, body, {"Location": location}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.body


class TestCappedGet:
    URL = "https://storage.googleapis.com/sen1floods11/x"

    def _check(self, url: str) -> str:
        return check_https_url(url, hosts=("storage.googleapis.com",))

    def test_connection_errors_are_transient_and_retried(self, monkeypatch) -> None:
        import requests

        calls = iter([requests.ConnectionError("reset"), requests.Timeout("slow"), _Response(200, b"ok")])

        def fake(url, **kw):
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(requests, "get", fake)
        waits = []
        assert capped_get(self.URL, check=self._check, cap=10, retries=3, sleep=waits.append) == b"ok"
        assert waits == [2.0, 4.0]

    def test_redirects_need_their_own_check(self, monkeypatch) -> None:
        import requests

        monkeypatch.setattr(requests, "get", lambda url, **kw: _Response(302, location="https://evil.example/x"))
        with pytest.raises(OSError, match="302"):
            capped_get(self.URL, check=self._check, cap=10)
        with pytest.raises(ValueError, match="refusing"):
            capped_get(self.URL, check=self._check, cap=10, redirect_check=self._check)

    def test_permanent_errors_are_not_retried(self, monkeypatch) -> None:
        import requests

        calls = []
        monkeypatch.setattr(requests, "get", lambda url, **kw: calls.append(url) or _Response(404))
        with pytest.raises(OSError, match="404") as info:
            capped_get(self.URL, check=self._check, cap=10, retries=5, sleep=lambda s: None)
        assert not isinstance(info.value, TransientHTTPError) and len(calls) == 1
