r"""Remote input, defensively: URL allow-lists, GDAL path guards, capped GETs, redaction.

Every byte this pipeline reads from the network comes from a public service whose responses
are treated as untrusted. Three rules, in one place:

- **A URL is checked with the parser that will make the request.** ``requests`` prepares URLs
  with urllib3; Python's ``urllib.parse`` disagrees with it on inputs such as
  ``https://evil.example\@allowed.host/`` (urllib3: host ``evil.example``). A URL passes only
  if both parsers agree on the host, it has no user-info, backslash, whitespace or control
  character, and its port is the default.
- **A local path given to GDAL is a plain path.** Driver prefixes (``GTIFF_DIR:``,
  ``NETCDF:``), ``file:`` forms other than ``file:///abs``, any ``/vsi*`` component in any case,
  and inline XML are refused: each lets a catalogue item make GDAL open other files or URLs.
- **Every transfer is bounded**: connect/read timeouts, a byte cap, no redirects except an
  explicitly allowed one, and retries only on transient failures.

:func:`redact` strips query strings (signing tokens) from URLs in any text, including the
``url: /path?...`` form urllib3 uses in its error messages.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

__all__ = ["HTTPStatusError", "TransientHTTPError", "capped_get", "check_https_url", "check_local_path", "redact",
           "redacted_traceback"]

LOGGER = logging.getLogger(__name__)

_FORBIDDEN_URL_CHARS = re.compile(r"[\\\s\x00-\x1f\x7f]")
_RETRY_STATUS = (429, 500, 502, 503, 504)


class TransientHTTPError(OSError):
    """A failure worth retrying: 429/5xx, a reset connection, a timeout."""


class HTTPStatusError(OSError):
    """A definite HTTP refusal (4xx other than 429), with its ``status``."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def check_https_url(url: str, *, hosts: Sequence[str] = (), suffixes: Sequence[str] = (),
                    path_prefixes: Sequence[str] = ()) -> str:
    """Return ``url`` if it is plain ``https`` to an allowed host (and path prefix); else raise."""
    from urllib3.util import parse_url

    text = str(url)
    if _FORBIDDEN_URL_CHARS.search(text):
        raise ValueError(f"refusing URL with backslash, whitespace or control characters: {redact(text)[:100]!r}")
    try:
        parsed, prepared = urlparse(text), parse_url(text)
        parsed.port  # noqa: B018 - urlparse validates the port lazily
    except ValueError as exc:  # urllib3's LocationParseError is a ValueError
        raise ValueError(f"refusing unparseable URL: {redact(text)[:100]!r} ({exc})") from None
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or prepared.scheme != "https":
        raise ValueError(f"refusing non-https URL: {redact(text)[:100]!r}")
    if parsed.username is not None or prepared.auth is not None or "@" in (parsed.netloc or ""):
        raise ValueError(f"refusing URL with user-info: {redact(text)[:100]!r}")
    if (prepared.host or "").lower() != host or not host:
        raise ValueError(f"refusing URL whose host parses ambiguously: {redact(text)[:100]!r}")
    if parsed.port not in (None, 443) or prepared.port not in (None, 443):
        raise ValueError(f"refusing URL with a non-default port: {redact(text)[:100]!r}")
    if not (host in {h.lower() for h in hosts} or any(host.endswith(s.lower()) for s in suffixes)):
        raise ValueError(f"refusing URL outside the allowed hosts: https://{host}")
    if path_prefixes:
        # A prefix only confines the path if the path cannot climb out of it: refuse dot segments, encoded dots or
        # slashes, and empty segments, all of which a client may normalise after this check.
        segments = parsed.path.split("/")[1:]
        if (any(seg in ("", ".", "..") for seg in segments[:-1]) or segments[-1:] in (["."], [".."])
                or re.search(r"%2e|%2f|%5c", parsed.path, re.IGNORECASE)):
            raise ValueError(f"refusing URL with a non-canonical path: https://{host}{parsed.path[:60]}")
        if not parsed.path.startswith(tuple(path_prefixes)):
            raise ValueError(f"refusing URL outside the allowed paths: https://{host}{parsed.path[:60]}")
    return text


def refuse_redirect(url: str, *, check: Callable[[str], str], timeout_s: float = 30.0) -> str:
    """``url`` after a ``HEAD`` that must answer without redirecting, before GDAL's ``/vsicurl`` (which follows
    redirects by itself and cannot be told not to) opens it.

    This narrows the gap rather than closing it: a server could answer the ``HEAD`` directly and redirect the ranged
    ``GET``s that follow (SECURITY.md).
    """
    import requests

    url = check(url)
    try:
        response = requests.head(url, timeout=(10, timeout_s), allow_redirects=False)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TransientHTTPError(f"HEAD {redact(url)[:100]}: {type(exc).__name__}") from None
    if 300 <= response.status_code < 400:
        raise ValueError(f"refusing a redirect before a GDAL read: {redact(url)[:100]} -> "
                         f"{redact(response.headers.get('Location', '?'))[:100]}")
    if response.status_code >= 400:
        raise HTTPStatusError(f"HEAD {redact(url)[:100]}: HTTP {response.status_code}", response.status_code)
    return url


def check_local_path(href: str) -> str:
    """Return the plain filesystem path GDAL may open for ``href``; refuse anything else."""
    text = str(href).strip()
    if text.startswith("file://"):
        text = text.removeprefix("file://")
        if not text.startswith("/"):
            raise ValueError(f"refusing file URL that is not file:///absolute: {href[:80]!r}")
    if text.lstrip().startswith("<"):
        raise ValueError(f"refusing inline XML as a path: {text[:60]!r}")
    if re.search(r"(^|/)vsi", text, re.IGNORECASE):
        raise ValueError(f"refusing GDAL virtual file system path: {text[:80]!r}")
    head = text.split("/", 1)[0]
    if ":" in head:
        raise ValueError(f"refusing driver-prefixed or scheme-prefixed path: {text[:80]!r}")
    if _FORBIDDEN_URL_CHARS.search(text.replace(" ", "")):
        raise ValueError(f"refusing path with control characters: {text[:80]!r}")
    return text


def redact(text: str) -> str:
    """Strip query strings (signing tokens) from every URL in ``text`` — for logs and errors."""
    out = re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\"]*", r"\1?<redacted>", str(text))
    return re.sub(r"(url: /[^\s?'\"]*)\?[^\s'\"]*", r"\1?<redacted>", out)  # urllib3 error text


def redacted_traceback(exc: BaseException) -> str:
    """The full traceback of ``exc`` as text, with every signing token removed (safe to log)."""
    import traceback

    return redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _read_capped(response: Any, cap: int, label: str) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_content(chunk_size=1 << 20):
        size += len(chunk)
        if size > cap:
            raise OSError(f"{label}: larger than {cap} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def capped_get(
    url: str,
    *,
    check: Callable[[str], str],
    cap: int,
    params: Mapping[str, str] | None = None,
    timeout_s: float = 120.0,
    redirect_check: Callable[[str], str] | None = None,
    retries: int = 1,
    backoff_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET ``url`` (validated by ``check``) with a byte cap and timeouts.

    No redirect is followed unless ``redirect_check`` validates the target, and then only one.
    ``retries`` > 1 retries transient failures with a doubling backoff.
    """
    for attempt in range(max(1, retries)):
        try:
            return _get_once(url, check=check, cap=cap, params=params, timeout_s=timeout_s,
                             redirect_check=redirect_check)
        except TransientHTTPError as exc:
            if attempt == max(1, retries) - 1:
                raise
            wait = backoff_s * 2**attempt
            LOGGER.warning("%s — retrying in %.0f s (%d/%d)", exc, wait, attempt + 1, retries - 1)
            sleep(wait)
    raise AssertionError("unreachable")


def _get_once(url: str, *, check: Callable[[str], str], cap: int, params: Mapping[str, str] | None,
              timeout_s: float, redirect_check: Callable[[str], str] | None) -> bytes:
    import requests

    label = redact(check(url)).split("?", 1)[0]
    try:
        with requests.get(url, params=params, timeout=(10, timeout_s), allow_redirects=False, stream=True) as r:
            if r.status_code in (301, 302, 303, 307, 308) and redirect_check is not None:
                target = redirect_check(r.headers.get("Location", ""))
                with requests.get(target, timeout=(10, timeout_s), allow_redirects=False, stream=True) as hop:
                    _raise_for(hop.status_code, label)
                    return _read_capped(hop, cap, label)
            _raise_for(r.status_code, label)
            return _read_capped(r, cap, label)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TransientHTTPError(f"{type(exc).__name__} for {label}: {redact(str(exc))[:200]}") from None


def _raise_for(status: int, label: str) -> None:
    if status == 200:
        return
    if status in _RETRY_STATUS:
        raise TransientHTTPError(f"HTTP {status} for {label}")
    raise HTTPStatusError(f"HTTP {status} for {label}", status)
