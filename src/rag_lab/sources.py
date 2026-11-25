"""Resolve an ingest source to a local file, downloading URLs into a cache."""

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

_SECONDS_PER_DAY = 24 * 60 * 60


def resolve_source(
    source: str,
    *,
    cache_dir: Path = Path("data/cache"),
    max_age_days: int = 30,
    max_retries: int = 2,
    timeout_seconds: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Return a local path for a filesystem path or an http(s) URL.

    An http(s) URL is saved at ``cache_dir / sha256(url) / filename`` when
    that file is missing, empty, or not strictly younger than ``max_age_days``.
    A path with no URL scheme is returned unchanged. ``max_retries`` is how
    many extra attempts follow a connection error or an HTTP 5xx.

    Raises ValueError for any other scheme, or when the download fails.
    A failed download does not replace an existing cache file.

    >>> resolve_source("data/sample/documents.jsonl").name
    'documents.jsonl'
    """
    scheme = urlparse(source).scheme
    if scheme == "":
        return Path(source)
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme '{scheme}' in '{source}'")
    destination = cache_dir / _cache_key(source) / _cache_name(source)
    if _is_fresh(destination, max_age_days):
        return destination
    body = _download(source, max_retries, timeout_seconds, transport)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _cache_key(url: str) -> str:
    """Return the SHA-256 hex digest used as the cache directory name.

    >>> len(_cache_key("https://example.test/a.parquet"))
    64
    """
    return hashlib.sha256(url.encode()).hexdigest()


def _cache_name(url: str) -> str:
    """Return the last URL path segment, or ``download`` when it has none.

    A path that ends in ``/`` has no filename.

    >>> _cache_name("https://example.test/a/corpus.parquet?x=1")
    'corpus.parquet'
    >>> _cache_name("https://example.test/a/")
    'download'
    """
    path = urlparse(url).path
    name = "" if path.endswith("/") else Path(path).name
    if name in {"", ".", ".."}:
        return "download"
    return name


def _is_fresh(path: Path, max_age_days: int) -> bool:
    """Return whether a cache file is non-empty and younger than max_age_days."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < max_age_days * _SECONDS_PER_DAY


def _download(
    url: str,
    max_retries: int,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> bytes:
    """Download a URL and return its body.

    Follows redirects. Retries connection errors and HTTP 5xx ``max_retries``
    extra times. Raises ValueError for any other status or when retries run out.
    """
    failure = "download failed"
    with httpx.Client(
        timeout=timeout_seconds, follow_redirects=True, transport=transport
    ) as client:
        for attempt in range(max_retries + 1):
            try:
                response = client.get(url)
            except httpx.TransportError as error:
                failure = str(error)
            else:
                if response.is_success:
                    return response.content
                failure = f"HTTP {response.status_code}"
                if response.status_code < 500:
                    break
            if attempt >= max_retries:
                break
    raise ValueError(f"Download failed for {url}: {failure}")
