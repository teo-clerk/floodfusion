r"""The frozen models' weights from Hugging Face Hub, each file checked against the freeze manifest.

The weights repository mirrors the run folders that ``reports/v12_freeze.json`` pins, at the same relative paths.
Per seed that is ``run.json``, ``best.pt``, ``eval_test.json`` and the SAR-only sibling's ``eval_test.json``
(:func:`frozen_files`). A downloaded file is kept only if its sha256 equals the manifest's, so the weights a live
map uses are byte for byte the ones the blind test froze. The revision must be a commit (40 hex digits), never a
branch, and ``best.pt`` is later loaded with ``weights_only=True`` (no code is unpickled).

Run::

    uv run --frozen python -m floodfusion.weights fetch --repo <owner>/<name> --revision <commit> \
        --run runs/v12/final/all/all/row21_fusion_context_seasonal/seed42 --dest .
    uv run --frozen python -m floodfusion.weights sums --run <run folder>      # SHA256SUMS for the upload

Torch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

__all__ = ["MANIFEST", "fetch", "frozen_files", "main", "sha256sums"]

LOGGER = logging.getLogger(__name__)

MANIFEST = Path("reports/v12_freeze.json")
_HOST = "huggingface.co"
#: Downloads redirect once: a large file to Hugging Face's CDN, a small one to a relative path on the Hub itself
#: (:func:`floodfusion.data.http.hub_redirect`, as for GEOID-Flood).
_CDN_SUFFIX = ".hf.co"
#: The largest frozen file is the fair TerraMind's checkpoint (428 MB).
_MAX_FILE_BYTES = 640 * 1024 * 1024
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}", re.ASCII)


def _sar_only_sibling(run: str) -> str | None:
    from floodfusion.case_study import SAR_ONLY_SIBLINGS

    path = PurePosixPath(run)
    sibling = SAR_ONLY_SIBLINGS.get(path.parent.name)
    return None if sibling is None else str(path.parent.parent / sibling / path.name / "eval_test.json")


def frozen_files(manifest: Path, run: str) -> dict[str, str]:
    """``{path: sha256}`` of the files a frozen run needs to predict, from the freeze manifest."""
    pinned = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
    run = str(PurePosixPath(run))
    wanted = [f"{run}/{name}" for name in ("run.json", "best.pt", "eval_test.json")]
    sibling = _sar_only_sibling(run)
    wanted += [sibling] if sibling else []
    missing = [p for p in wanted if p not in pinned]
    if missing:
        raise KeyError(f"{run}: {missing} not pinned by {manifest}")
    return {p: pinned[p] for p in sorted(wanted)}


def _check(repo: str, revision: str, path: str) -> PurePosixPath:
    if not _REPO.fullmatch(repo):
        raise ValueError(f"not a Hugging Face repository id: {repo[:100]!r}")
    if not _COMMIT.fullmatch(revision):
        raise ValueError(f"the revision must be a pinned commit (40 hex digits), not {revision[:60]!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"refusing a weights path outside the destination: {path[:200]!r}")
    return pure


def _http_get(url: str) -> bytes:
    from floodfusion.data.http import capped_get, check_https_url, hub_redirect

    return capped_get(url, check=lambda u: check_https_url(u, hosts=(_HOST,)), cap=_MAX_FILE_BYTES, retries=3,
                      timeout_s=600.0,
                      redirect_check=lambda loc: hub_redirect(url, loc, host=_HOST, cdn_suffix=_CDN_SUFFIX))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(files: Mapping[str, str], dest: Path, *, repo: str, revision: str,
          get: Callable[[str], bytes] | None = None) -> list[Path]:
    """Download ``files`` (``{path: sha256}``) from ``repo`` at ``revision`` into ``dest``; return their paths.

    A file already in ``dest`` with the right sha256 is not fetched again; a download with another sha256 raises
    and is not written.
    """
    from floodfusion.data.records import atomic_write_bytes

    get = get or _http_get
    out = []
    for path, expected in sorted(files.items()):
        pure = _check(repo, revision, path)
        target = Path(dest).joinpath(*pure.parts)
        if not (target.is_file() and _sha256_file(target) == expected):
            blob = get(f"https://{_HOST}/{repo}/resolve/{revision}/{pure}")
            actual = hashlib.sha256(blob).hexdigest()
            if actual != expected:
                raise ValueError(f"{path}: sha256 {actual[:12]}… is not the frozen {expected[:12]}…")
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target, blob)
            LOGGER.info("fetched %s (%d bytes)", path, len(blob))
        out.append(target)
    return out


def sha256sums(files: Mapping[str, str]) -> str:
    """The ``SHA256SUMS`` text (``sha256sum`` format) of ``files``, for the weights repository."""
    return "".join(f"{sha}  {path}\n" for path, sha in sorted(files.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The frozen weights from Hugging Face Hub, checked file by file.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "sums"):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", required=True, help="a frozen run folder (repeatable)")
        p.add_argument("--manifest", type=Path, default=MANIFEST)
    sub.choices["fetch"].add_argument("--repo", required=True)
    sub.choices["fetch"].add_argument("--revision", required=True, help="a commit of the weights repository")
    sub.choices["fetch"].add_argument("--dest", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    files = {p: s for run in args.run for p, s in frozen_files(args.manifest, run).items()}
    if args.cmd == "sums":
        print(sha256sums(files), end="")  # noqa: T201 - the command's output is the SHA256SUMS text
        return 0
    fetch(files, args.dest, repo=args.repo, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
