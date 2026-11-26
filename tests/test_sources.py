import hashlib
import os
import time
from pathlib import Path

import httpx
import pytest

from rag_lab.settings import Settings
from rag_lab.sources import resolve_source

URL = "https://example.test/a/corpus.parquet"
DAY = 24 * 60 * 60


def test_local_path_is_returned_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "docs.parquet"

    assert resolve_source(str(source), cache_dir=tmp_path / "cache") == source


def test_download_is_cached_by_url(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == URL
        return httpx.Response(200, content=b"one")

    path = resolve_source(
        URL, cache_dir=tmp_path, transport=httpx.MockTransport(handle)
    )

    assert path == (
        tmp_path / hashlib.sha256(URL.encode()).hexdigest() / "corpus.parquet"
    )
    assert path.read_bytes() == b"one"


def test_fresh_cache_is_not_downloaded_again(tmp_path: Path) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"one")

    first = resolve_source(
        URL, cache_dir=tmp_path, transport=httpx.MockTransport(handle)
    )

    def fail(_: httpx.Request) -> httpx.Response:
        raise AssertionError("fresh cache must not be downloaded")

    second = resolve_source(
        URL, cache_dir=tmp_path, transport=httpx.MockTransport(fail)
    )

    assert second == first
    assert second.read_bytes() == b"one"


def test_cache_older_than_max_age_is_downloaded_again(tmp_path: Path) -> None:
    _seed(tmp_path, b"old", age_days=31)

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"new")

    path = resolve_source(
        URL, cache_dir=tmp_path, max_age_days=30, transport=httpx.MockTransport(handle)
    )

    assert path.read_bytes() == b"new"


def test_empty_cache_file_is_downloaded_again(tmp_path: Path) -> None:
    _seed(tmp_path, b"", age_days=0)

    path = resolve_source(
        URL,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"new")),
    )

    assert path.read_bytes() == b"new"


def test_cache_exactly_max_age_is_downloaded_again(tmp_path: Path) -> None:
    _seed(tmp_path, b"old", age_days=30)

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"new")

    path = resolve_source(
        URL, cache_dir=tmp_path, max_age_days=30, transport=httpx.MockTransport(handle)
    )

    assert path.read_bytes() == b"new"


def test_query_string_is_not_part_of_the_filename(tmp_path: Path) -> None:
    url = "https://example.test/a/corpus.parquet?download=1"

    path = resolve_source(
        url,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x")),
    )

    assert path.name == "corpus.parquet"
    assert path.parent.name == hashlib.sha256(url.encode()).hexdigest()


def test_url_without_a_filename_uses_download(tmp_path: Path) -> None:
    path = resolve_source(
        "https://example.test/a/",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x")),
    )

    assert path.name == "download"


def test_non_http_scheme_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ftp"):
        resolve_source("ftp://example.test/a.parquet", cache_dir=tmp_path)


def test_redirect_stores_the_final_body(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                302, headers={"location": "https://files.test/corpus.parquet"}
            )
        return httpx.Response(200, content=b"final")

    path = resolve_source(
        "https://example.test/start",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handle),
    )

    assert path.read_bytes() == b"final"


def test_http_404_does_not_replace_or_create_a_cache_file(tmp_path: Path) -> None:
    seeded = _seed(tmp_path, b"keep", age_days=31)
    transport = httpx.MockTransport(lambda _: httpx.Response(404))

    with pytest.raises(ValueError, match="404"):
        resolve_source(URL, cache_dir=tmp_path, transport=transport)

    assert seeded.read_bytes() == b"keep"
    assert set(tmp_path.rglob("*")) == {seeded.parent, seeded}

    empty = tmp_path / "empty"
    with pytest.raises(ValueError, match="404"):
        resolve_source(
            "https://example.test/missing.parquet",
            cache_dir=empty,
            transport=transport,
        )
    assert not empty.exists()


def test_server_error_is_retried_then_saved(tmp_path: Path) -> None:
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    path = resolve_source(
        URL,
        cache_dir=tmp_path,
        max_retries=2,
        transport=httpx.MockTransport(handle),
    )

    assert calls == 3
    assert path.read_bytes() == b"ok"


def test_exhausted_server_error_leaves_no_cache_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="503"):
        resolve_source(
            URL,
            cache_dir=tmp_path,
            max_retries=0,
            transport=httpx.MockTransport(lambda _: httpx.Response(503)),
        )

    assert list(tmp_path.rglob("*")) == []


def test_cache_max_age_defaults_to_30_days() -> None:
    assert Settings(_env_file=None).cache_max_age_days == 30  # type: ignore[call-arg]


def _seed(cache_dir: Path, body: bytes, age_days: float) -> Path:
    path = cache_dir / hashlib.sha256(URL.encode()).hexdigest() / "corpus.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    timestamp = time.time() - age_days * DAY
    os.utime(path, (timestamp, timestamp))
    return path
