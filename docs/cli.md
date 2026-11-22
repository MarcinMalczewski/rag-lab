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
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup naive-100-0
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup naive-500-100
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

`query`, `status` and `reset` are placeholders for now.

## Commands

The CLI operates on a small set of setups and ingestion templates predefined in
code. A setup fixes chunking, embedding, retrieval, and Qdrant indexing choices;
a template converts one input format into normalized documents. Currently,
`raglab setups`, `raglab templates`, and the document-loading part of
`raglab ingest` (documents are stored in PostgreSQL) are implemented; the rest is placeholder.

```bash
raglab setups
```

Lists supported setup IDs and their main settings. Use it to discover valid
values for `--setup`.

```bash
raglab templates
```

Lists ingestion template IDs; use it to discover valid values for `--template`.
Only `jsonl-documents` exists so far.

```bash
raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup naive-100-0
```

Currently loads and validates the documents (reporting file and line on errors),
then saves them to PostgreSQL under the source checksum as the corpus version
and records a run for the setup (status `documents_stored`). Re-ingesting the
same source does not duplicate documents. It also chunks the documents in memory
with the setup's chunking profile and prints chunk counts per document; the
chunks are not saved yet. Embedding and indexing are not implemented yet. The intended full behavior is:

```bash
raglab ingest data/squad --template squad-contexts --setup naive-500-100
```

Normalizes input documents, then chunks and embeds them before writing vectors
and setup metadata to Qdrant. Store the setup ID, resolved parameters, template
ID, and source/corpus version with every ingestion run.

Initial code-defined templates can include `msmarco-passages`, `squad-contexts`,
`markdown-directory`, and `jsonl-documents`. For SQuAD, index unique contexts
and retain questions/answer spans for evaluation rather than treating questions
as source documents.

```bash
raglab query "What is late chunking?" --setup naive-500-100
```

Embeds the query and searches only vectors indexed for the selected setup.

```bash
raglab status
```

Shows each setup's indexed document/vector counts, target collection, last
ingestion time, and readiness state.

```bash
raglab reset --setup naive-500-100
```

Removes only vectors and index metadata for the selected setup, allowing a clean
re-index. It should request confirmation; `--yes` may bypass it for automation.

