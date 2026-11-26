# Parquet documents implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest a parquet table of documents from a local path or an HTTP URL, with a shared downloader other templates can reuse.

**Architecture:** `resolve_source` turns an `http`/`https` URL into a file under `data/cache/` and returns a local path unchanged. `ingest` calls it before `template.load` and `source_checksum`. `ParquetDocumentsTemplate` reads that file with pyarrow. One row is one document.

**Tech Stack:** Python 3.11, pyarrow, httpx, pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-parquet-documents-design.md`

## Global Constraints

- Registered template id is `parquet-documents`. Defaults: `id_field="_id"`, `text_field="text"`, `title_field="title"`.
- No CLI flags to rename columns.
- Cache path is `data/cache/<sha256 of the exact URL string>/<filename>`.
- Filename is the last URL path segment, or `download` when that segment is empty, `.`, or `..`.
- Reuse a cached file whose age is strictly less than `cache_max_age_days` (default 30). Age is `st_mtime` minus now. Older files are downloaded again.
- Do not use ETag, Last-Modified, or a remote checksum.
- Follow redirects.
- Retry connection errors and HTTP 5xx `model_max_retries` times after the first attempt (same meaning as `OllamaEmbedder.max_retries`). Any other HTTP status raises `ValueError` and includes the status code. Exhausted retries raise `ValueError`.
- A failed download must not replace an existing cache file and must not leave a partial file as the cache.
- Corpus version stays `source_checksum` of the local file bytes.
- `pyarrow` is a main dependency. Tests do not use the network.
- Extra parquet columns are ignored. `metadata` is `{}`.
- Do not coerce non-strings. Null or blank title becomes `None`.

## Review Focus

- An `ftp://` or other non-http scheme is rejected with `ValueError`, not treated as a local path. Test in Task 1.
- A query string is not part of the cache filename (`.../corpus.parquet?download=1` caches as `corpus.parquet`). Test in Task 1.
- A 302 to another `https` URL stores the final body. Test in Task 1.
- HTTP 404, and HTTP 500 after retries, leave an existing fresh cache file unchanged and do not create a cache file when there was none. Test in Task 1.
- A numeric `title` raises `ValueError` naming the row. Test in Task 2.

---

### Task 1: Shared source resolver

**Files:**
- Create: `src/rag_lab/sources.py`
- Create: `tests/test_sources.py`
- Modify: `src/rag_lab/settings.py`
- Modify: `.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: `resolve_source(source: str, *, cache_dir: Path = Path("data/cache"), max_age_days: int = 30, max_retries: int = 2, timeout_seconds: float = 60.0, transport: httpx.BaseTransport | None = None) -> Path`
- Produces: `Settings.cache_max_age_days: int = 30`

- [ ] **Step 1: Write the failing tests**

`tests/test_sources.py` uses `httpx.MockTransport` and `tmp_path`. Assertions:

- `resolve_source(str(path), cache_dir=tmp_path)` returns `Path(str(path))` and performs no HTTP.
- A 200 response stores the body at `cache_dir / sha256(url) / "corpus.parquet"`.
- A second call with a transport that raises is not called and returns the same path.
- A file with `st_mtime` 31 days ago is replaced by the new body.
- A file with `st_mtime` exactly 30 days ago is replaced (age is not strictly younger).
- `https://example.test/a/corpus.parquet?download=1` uses filename `corpus.parquet`.
- `https://example.test/a/` uses filename `download`.
- `ftp://example.test/a.parquet` raises `ValueError`.
- First response 302 `Location: https://files.test/corpus.parquet`, second 200: saved body is the 200 body.
- 404 raises `ValueError` matching `404`. With a pre-seeded cache file, that file's bytes are unchanged and no extra file remains.
- Two 503 responses then 200, with `max_retries=2`, returns the 200 body.
- `max_retries=0` and a 503 raises `ValueError`. No cache file remains.
- `Settings(_env_file=None).cache_max_age_days == 30`.

- [ ] **Step 2: Run `uv run pytest tests/test_sources.py -v`**

Expected: FAIL on import of `rag_lab.sources` or missing `cache_max_age_days`.

- [ ] **Step 3: Implement**

Add `cache_max_age_days: int = 30` to `Settings`. Document it in `.env.example` as `CACHE_MAX_AGE_DAYS=30`. Add `data/cache/` to `.gitignore`.

`resolve_source` writes to a sibling temporary file in the cache directory and `os.replace`s it onto the destination only after the full body is read. `httpx.Client(..., follow_redirects=True, transport=transport)`.

- [ ] **Step 4: Run `uv run pytest tests/test_sources.py -v`**

Expected: PASS

### Task 2: Parquet template

**Files:**
- Modify: `src/rag_lab/templates.py`
- Modify: `tests/test_templates.py`
- Modify: `pyproject.toml` (add `pyarrow`)
- Modify: `uv.lock` via `uv add pyarrow`

**Interfaces:**
- Consumes: `Document`
- Produces: `ParquetDocumentsTemplate` with defaults `_id`, `text`, `title`, registered in `TEMPLATES`

- [ ] **Step 1: Write the failing tests** in `tests/test_templates.py`

Build tables with `pyarrow.parquet.write_table`. Assertions:

- Registry order is `["jsonl-documents", "parquet-documents"]`.
- Two rows plus an extra `url` column: ids, texts, titles match; `metadata == {}`.
- Title `""` and title `None` both become `None`.
- Duplicate `_id` on row 2 raises `ValueError` matching `:2:` and `duplicate id`.
- Missing `title` column raises `ValueError` naming `title`.
- Empty table raises `ValueError` matching `no documents found`.
- Integer `_id` raises `ValueError` matching `:1:` and `_id`.
- Integer `title` raises `ValueError` matching `:1:` and `title`.
- Text `"  "` raises `ValueError` matching `text`.

- [ ] **Step 2: Run `uv run pytest tests/test_templates.py -v`**

Expected: FAIL because `parquet-documents` is unknown or the class is missing.

- [ ] **Step 3: Implement `ParquetDocumentsTemplate.load`**

Read with `pyarrow.parquet.read_table`. Row numbers start at 1. Require the three column names. Reject non-strings and blank `_id` / `text`. Blank title is `None`.

- [ ] **Step 4: Run `uv run pytest tests/test_templates.py -v`**

Expected: PASS

### Task 3: Ingest wiring and docs

**Files:**
- Modify: `src/rag_lab/cli.py` (`_ingest`)
- Modify: `tests/test_cli.py`
- Modify: `docs/cli.md`, `docs/rag-strategy.md`, `docs/plan.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `resolve_source`, `Settings.cache_max_age_days`, `Settings.model_max_retries`, `Settings.model_request_timeout_seconds`, `get_template("parquet-documents")`

- [ ] **Step 1: Write the failing CLI test**

`test_ingest_parquet_url_uses_cache` uses the existing Postgres and Qdrant fixtures and `FakeEmbedder`. Monkeypatch `rag_lab.cli.resolve_source` so it records `source` and `max_age_days` and returns a one-row parquet. Run `main(["ingest", "https://example.test/corpus.parquet", "--template", "parquet-documents", "--setup", "nomic-naive-100-0"])`.

Assert exit 0, output contains the URL and the parquet path, `max_age_days` equals `get_settings().cache_max_age_days`, and `store.load_documents` returns that one document.

Update `test_templates_command_lists_templates` to assert `parquet-documents` is listed.

- [ ] **Step 2: Run `uv run pytest tests/test_cli.py::test_ingest_parquet_url_uses_cache tests/test_cli.py::test_templates_command_lists_templates -v`**

Expected: FAIL because ingest does not call `resolve_source` or the output has no cache path.

- [ ] **Step 3: Call `resolve_source` in `_ingest`**

Print `Source:` as the argument the user gave. When the returned path is not `Path(argument)`, print `Cached:   {path}` before loading. Pass `cache_max_age_days`, `model_max_retries`, and `model_request_timeout_seconds` from settings.

- [ ] **Step 4: Docs**

`docs/cli.md`: `templates` lists `jsonl-documents` and `parquet-documents`. Add the nfcorpus URL example under Ingestion. Say a URL is cached for `CACHE_MAX_AGE_DAYS` (default 30) and a local parquet path uses the same template.

`docs/rag-strategy.md`: add a `parquet-documents` row (one row per parquet record; `_id`, `text`, `title`).

`docs/plan.md`: mark a parquet-documents item done under ingestion templates.

`CLAUDE.md`: mention `parquet-documents` and that `ingest` accepts a local path or an `http`/`https` URL.

- [ ] **Step 5: Run `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`**

Expected: PASS

---

## Self-review

- Spec coverage: template, URL and local path, shared helper, 30-day cache, retries, redirects, checksum unchanged, docs, no network in tests. No remote validators.
- The ingest test pins that the CLI passes `max_age_days` through. Download behavior itself is Task 1.
- Commit steps are omitted. This checkout commits only when the user asks.
