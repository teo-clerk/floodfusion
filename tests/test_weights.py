"""The frozen weights from Hugging Face Hub (``floodfusion.weights``), offline: every file checked against the freeze."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from floodfusion import weights

ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/v12/final/all/all/row21_fusion_context_seasonal/seed42"
REV = "0123456789abcdef0123456789abcdef01234567"


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def test_a_run_needs_its_three_files_and_its_sar_only_calibration() -> None:
    files = weights.frozen_files(ROOT / "reports" / "v12_freeze.json", RUN)
    assert sorted(files) == [f"{RUN}/best.pt", f"{RUN}/eval_test.json", f"{RUN}/run.json",
                             "runs/v12/final/all/all/row21_fusion_context_seasonal__eval_sar_only/seed42/eval_test.json"]
    assert all(len(v) == 64 for v in files.values())


def test_a_run_the_freeze_does_not_pin_is_refused() -> None:
    with pytest.raises(KeyError, match="not pinned"):
        weights.frozen_files(ROOT / "reports" / "v12_freeze.json", "runs/v12/final/all/all/other/seed42")


def test_fetch_downloads_each_file_from_the_pinned_revision_and_checks_it(tmp_path: Path) -> None:
    blobs = {"a/run.json": b"{}", "a/best.pt": b"weights"}
    urls = []

    def get(url: str) -> bytes:
        urls.append(url)
        return blobs[url.split(f"/resolve/{REV}/", 1)[1]]

    files = {path: _sha(blob) for path, blob in blobs.items()}
    written = weights.fetch(files, tmp_path, repo="someone/floodfusion", revision=REV, get=get)
    assert sorted(urls) == [f"https://huggingface.co/someone/floodfusion/resolve/{REV}/a/best.pt",
                            f"https://huggingface.co/someone/floodfusion/resolve/{REV}/a/run.json"]
    assert all((tmp_path / p).read_bytes() == b for p, b in blobs.items()) and len(written) == 2
    weights.fetch(files, tmp_path, repo="someone/floodfusion", revision=REV, get=get)
    assert len(urls) == 2                                   # files already verified are not fetched again


def test_a_file_that_is_not_the_frozen_one_is_refused_and_not_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not the frozen"):
        weights.fetch({"a/best.pt": _sha(b"frozen")}, tmp_path, repo="someone/floodfusion", revision=REV,
                      get=lambda url: b"something else")
    assert not (tmp_path / "a" / "best.pt").exists()


@pytest.mark.parametrize(("repo", "revision", "path"), [
    ("someone/floodfusion", "main", "a/best.pt"),            # a branch moves: only a commit is pinned
    ("not a repo", REV, "a/best.pt"),
    ("someone/floodfusion", REV, "../escape.pt"),
    ("someone/floodfusion", REV, "/etc/passwd"),
])
def test_fetch_refuses_an_unpinned_revision_a_bad_repo_or_an_escaping_path(tmp_path, repo, revision, path) -> None:
    with pytest.raises(ValueError):
        weights.fetch({path: "0" * 64}, tmp_path, repo=repo, revision=revision, get=lambda url: b"")


def test_sha256sums_lists_every_file_for_the_upload(tmp_path: Path) -> None:
    files = {"a/run.json": _sha(b"{}"), "a/best.pt": _sha(b"w")}
    assert weights.sha256sums(files) == f"{files['a/best.pt']}  a/best.pt\n{files['a/run.json']}  a/run.json\n"


def test_the_sums_command_prints_the_frozen_hashes(capsys) -> None:
    assert weights.main(["sums", "--manifest", str(ROOT / "reports" / "v12_freeze.json"), "--run", RUN]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 and all(len(line.split("  ")[0]) == 64 for line in lines)
    assert lines[0].endswith(f"{RUN}/best.pt")


class _Response:
    def __init__(self, status: int, body: bytes = b"", location: str = "") -> None:
        self.status_code, self.body, self.headers = status, body, {"Location": location}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.body


@pytest.mark.parametrize(("location", "hop"), [
    ("/api/resolve-cache/models/o/r/x.json", "https://huggingface.co/api/resolve-cache/models/o/r/x.json"),
    ("https://us.aws.cdn.hf.co/xet-bridge-us/abc", "https://us.aws.cdn.hf.co/xet-bridge-us/abc"),
])
def test_the_download_follows_the_hubs_redirect_for_small_and_large_files(monkeypatch, location, hop) -> None:
    # 2026-11-12: the first public fetch of the weights failed on the relative redirect Hugging Face gives small files.
    import requests

    seen: list[str] = []

    def fake(url, **kw):
        seen.append(url)
        return _Response(307, location=location) if len(seen) == 1 else _Response(200, b"{}")

    monkeypatch.setattr(requests, "get", fake)
    url = f"https://huggingface.co/o/r/resolve/{REV}/x.json"
    assert weights._http_get(url) == b"{}"
    assert seen == [url, hop]
