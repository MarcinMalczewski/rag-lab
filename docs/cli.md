# CLI

## Quick start (copy and paste)

```bash
uv sync                # install dependencies (once)
./start.sh             # start Qdrant and PostgreSQL; creates .env if missing
```

```bash
uv run raglab setups
uv run raglab templates
```

Ingest the sample documents (needs `./start.sh` first):

```bash
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-500-100
```

Stop the services and follow logs:

```bash
docker compose --env-file .env -f docker-compose.yml down
docker compose --env-file .env -f docker-compose.yml logs -f qdrant postgres
```

Tests and checks (tests need Docker):

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Query after ingesting (needs a complete run for that setup):

```bash
uv run raglab query "What is late chunking?" --setup nomic-naive-500-100 --top-k 5
```

`status` and `reset` are placeholders for now.

## Commands

The CLI operates on a small set of setups and ingestion templates predefined in
code. A setup fixes chunking, embedding, retrieval, and Qdrant indexing choices;
a template converts one input format into normalized documents. Currently,
`raglab setups`, `raglab templates`, `raglab ingest` (documents, chunks and
vectors) and `raglab query` are implemented; `status` and `reset` are placeholders.

```bash
raglab setups
```

Lists supported setup IDs and their main settings. A setup ID is an embedding
profile plus a chunking profile (`nomic`/`minilm` + `naive-100-0`/`naive-500-100`),
generated from the registries in `setups.py`. Use it to discover valid
values for `--setup`.

```bash
raglab templates
```

Lists ingestion template IDs; use it to discover valid values for `--template`.
Only `jsonl-documents` exists so far.

```bash
raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
```

Currently loads and validates the documents (reporting file and line on errors),
then saves them to PostgreSQL under the source checksum as the corpus version
and records a run for the setup (status `documents_stored`). Re-ingesting the
same source does not duplicate documents. It also chunks the documents in memory
with the setup's chunking profile, embeds the chunks with the setup's embedding profile: `nomic` uses Ollama
(needs `ollama pull nomic-embed-text`), `minilm` runs locally through Hugging
Face (needs `uv sync --extra hf`) and writes them to a Qdrant collection named
after the embedding model and dimensions (e.g. `nomic-embed-text-768`). Points
carry `setup_id`, `run_id`, `document_id`, offsets and the chunk text in the payload; their IDs
are recorded in PostgreSQL and the run ends as `complete`, or `failed` if
embedding or indexing errors. Setups whose chunking is not implemented
(`late-chunking-v1`) only store the documents. The intended full behavior is:

```bash
raglab ingest data/squad --template squad-contexts --setup nomic-naive-500-100
```

Normalizes input documents, then chunks and embeds them before writing vectors
and setup metadata to Qdrant. Store the setup ID, resolved parameters, template
ID, and source/corpus version with every ingestion run.

Initial code-defined templates can include `msmarco-passages`, `squad-contexts`,
`markdown-directory`, and `jsonl-documents`. For SQuAD, index unique contexts
and retain questions/answer spans for evaluation rather than treating questions
as source documents.

```bash
raglab query "What is late chunking?" --setup nomic-naive-500-100
```

Embeds the query with the setup's embedding profile and searches only the
selected setup's vectors from its latest complete run. `--top-k` (default 5)
sets the number of results. Each result shows rank, cosine score, document,
title, character range and a text preview; the text and metadata are read from
PostgreSQL by document ID and offsets, not from the Qdrant payload. A setup
without a complete run (never ingested, or ingest failed) is reported as an
error. Query prefixes (`search_query:`) are not applied yet.

```bash
raglab status
```

Shows each setup's indexed document/vector counts, target collection, last
ingestion time, and readiness state.

```bash
raglab reset --setup nomic-naive-500-100
```

Removes only vectors and index metadata for the selected setup, allowing a clean
re-index. It should request confirmation; `--yes` may bypass it for automation.

