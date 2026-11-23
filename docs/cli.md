# CLI

The CLI works with a small set of setups and ingestion templates defined in
code. A **setup** is an embedding profile plus a chunking profile (and the
retrieval mode); a **template** converts one input format into normalized
documents. All examples below use the sample data in `data/sample/`, the
`jsonl-documents` template and the `nomic-naive-100-0` setup.

Sections: [Setup](#setup) · [Discovering setups and templates](#discovering-setups-and-templates) ·
[Ingestion](#ingestion) · [Querying](#querying) · [Evaluation](#evaluation) ·
[Status and reset](#status-and-reset) · [Development](#development)

Typical flow (copy and paste):

```bash
uv sync
./start.sh
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
uv run raglab query "What is late chunking?" --setup nomic-naive-100-0
uv run raglab evaluate --setup nomic-naive-100-0 --questions data/sample/questions.jsonl
```

## Setup

```bash
uv sync                # install dependencies (once)
./start.sh             # start Qdrant and PostgreSQL; creates .env if missing
```

- `nomic-*` setups embed through Ollama: pull the model once with
  `ollama pull nomic-embed-text` and set `OLLAMA_URL` in `.env` if Ollama is not
  on `localhost` (for example Windows Ollama reached from WSL).
- `minilm-*` setups embed locally with Hugging Face: run `uv sync --extra hf`.
  A plain `uv sync` or `uv run` removes those packages again; use
  `uv run --extra hf ...` to keep them.

Stop the services and follow their logs:

```bash
docker compose --env-file .env -f docker-compose.yml down
docker compose --env-file .env -f docker-compose.yml logs -f qdrant postgres
```

## Discovering setups and templates

```bash
uv run raglab setups
uv run raglab templates
```

`setups` lists the valid values for `--setup`. A setup ID is an embedding
profile plus a chunking profile (`nomic`/`minilm` + `naive-100-0`/`naive-500-100`),
generated from the registries in `setups.py`. `late-chunking-v1` is reserved and
not implemented. `templates` lists the valid values for `--template`; only
`jsonl-documents` exists so far.

## Ingestion

```bash
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
```

Needs: `./start.sh`, and the setup's embedding model (see [Setup](#setup)).

Loads the documents with the template, saves them to PostgreSQL, chunks them,
embeds the chunks with the setup's embedding profile and indexes them in Qdrant,
recording the run. Each step is printed. Re-ingesting the same source does not
duplicate anything. Setups whose chunking is not implemented (`late-chunking-v1`)
only store the documents. More templates (`msmarco-passages`, `squad-contexts`,
`markdown-directory`) are planned.

## Querying

```bash
uv run raglab query "What is late chunking?" --setup nomic-naive-100-0 --top-k 5
```

Needs: a complete run for the setup (ingest first).

Searches only that setup's vectors and prints the `--top-k` (default 5) best
passages with score, document, character range and a text preview. Query prefixes
(`search_query:`) are not applied yet.

## Evaluation

```bash
uv run raglab evaluate --setup nomic-naive-100-0 --questions data/sample/questions.jsonl --top-k 5
```

Needs: a complete run for the setup, and a questions file whose labels match the
ingested corpus.

Runs every labelled question against the setup and scores the retrieved chunks
against the labelled answer spans (recall, precision, first relevant rank, hit
rate, MRR, nDCG). Run it once per setup to compare them. How it works and how to
read the scores: [Evaluation](evaluation.md).

## Status and reset

Not implemented yet; both commands only validate the setup and print a message.

```bash
uv run raglab status
uv run raglab reset --setup nomic-naive-100-0
```

`status` is meant to show each setup's indexed document and vector counts, target
collection, last ingestion time and readiness. `reset` is meant to remove only
the vectors and index metadata of the selected setup, so it can be re-indexed
cleanly. It should ask for confirmation, and `--yes` may skip it for automation.

## Development

```bash
uv run --extra hf pytest        # tests start Postgres and Qdrant containers (Docker required)
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
