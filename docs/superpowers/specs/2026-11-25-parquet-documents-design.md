# Parquet documents template

## Intent

Ingest a table of documents from a parquet file, local or downloaded from a URL, then query it with the existing setup path. The first file is the BeIR nfcorpus corpus parquet. The downloader is shared so later templates can take a URL too.

## Goals

- Register `parquet-documents` next to `jsonl-documents`.
- Expect columns `_id`, `text`, and `title`. One row is one document.
- `raglab ingest` accepts a local path or an `http`/`https` URL for this template.
- Download through one helper that any later template can use. The helper is not parquet-specific.
- Document the command in the existing CLI docs.

## Non-goals

- CLI flags to rename columns. A different column layout is a different template instance in code, same as `JsonlDocumentsTemplate`.
- Nested parquet values, dataset shards (many files, one URL), qrels, or evaluation questions for nfcorpus.
- A named corpus such as `--corpus nfcorpus`.

## Approaches

1. **URL or path into `ingest`, parquet template for the table.** Chosen. The URL is an example of a source, and templates here are formats.
2. **Named corpus with the URL fixed in code.** Rejected. A new dataset would need a code change, and the request is a generic tool.
3. **Column flexibility.** Chosen level: defaults `_id`, `text`, `title` on the registered template. Other names mean changing that template. Hardcoding with no constructor fields was tighter than JSONL. Accepting any nested parquet was more tool than this file needs.

Download placement: the template reads a local file, as `load(source: Path)` does today. `ingest` turns a URL into a file first, via the shared helper. Every template then gains URL support without its own HTTP code. Putting HTTP inside `ParquetDocumentsTemplate` would make this template special and force the next template to copy it.

## Behavior

Registered template: `ParquetDocumentsTemplate` with `id_field="_id"`, `text_field="text"`, `title_field="title"`.

`load` uses `pyarrow` and returns `list[Document]`.

- The three columns must exist. A missing column raises `ValueError` and names the column.
- `_id` and `text` must be non-empty strings. A blank or non-string value raises `ValueError` with the row number (first row is 1).
- A null or blank `title` becomes `None`. Any other title is stored as the string.
- Duplicate `_id` values raise `ValueError` with the row number.
- An empty table raises `ValueError` (`no documents found`).
- Other columns are ignored. `metadata` stays `{}`.
- Non-string `_id`, `text`, or `title` values are rejected. Call: do not coerce numbers. A file with different types needs a different template.

Shared helper, used by `ingest` before `template.load` and `source_checksum`:

- A local path is returned as a `Path`.
- An `http` or `https` URL is downloaded into `data/cache/<sha256 of the URL>/<filename>`.
- The filename is the last URL path segment, or `download` when the URL has none.
- A cached file younger than `cache_max_age_days` (default 30, from settings) is reused. Age is the file's modification time, set when the download finishes. An older file is downloaded again and replaces the cache. Deleting the file also forces a new download.
- Remote change detection is not used. A checksum of the local file cannot show that the origin changed, and ETag or Last-Modified headers are not reliable on arbitrary URLs.
- Redirects are followed (the Hugging Face `resolve` URL redirects).
- Connection errors and HTTP 5xx are retried with `model_max_retries` from settings. Other HTTP errors raise `ValueError` with the status.
- The corpus version stays the SHA-256 of the local file bytes.
- `ingest` prints the source the user gave. When that source is a URL, it also prints the cache path.

`data/cache/` is gitignored.

Example:

```bash
uv run raglab ingest \
  https://huggingface.co/datasets/BeIR/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet \
  --template parquet-documents \
  --setup nomic-naive-100-0
```

A local `.parquet` path uses the same template and the same command shape.

## Docs

Extend `docs/cli.md` (template list and this ingest example), add a row to the template table in `docs/rag-strategy.md`, and add the milestone in `docs/plan.md`. Mention the template and URL source in `CLAUDE.md`.

## Tests

No network.

- Build a tiny parquet with `pyarrow` and cover a good file, a blank title, a duplicate id, a missing column, an empty table, and a non-string id.
- The downloader uses `httpx.MockTransport`, the same pattern as the embedding tests: a saved file, a second call within 30 days that does not request again, a file older than 30 days that is downloaded again, and a failed status.

## Builder calls

- Dependency: `pyarrow` on the main dependency list. `httpx` is already there.
- Cache directory is `data/cache` under the current working directory. `cache_max_age_days` defaults to 30 in settings.
- Extra columns are ignored rather than copied into metadata.
- All rows are loaded into memory, same as JSONL.
