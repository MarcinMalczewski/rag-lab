# CLI

The CLI works with a small set of setups and ingestion templates defined in
code. A **setup** is an embedding profile plus a chunking profile (and the
retrieval mode); a **template** converts one input format into normalized
documents. The examples use the `nomic-naive-100-0` setup. Each corpus example
ingests one source and then queries that source.

Sections: [Setup](#setup) · [Discovering setups and templates](#discovering-setups-and-templates) ·
[Corpus examples](#corpus-examples) · [Ingestion](#ingestion) · [Querying](#querying) ·
[Evaluation](#evaluation) · [Status and reset](#status-and-reset) · [Development](#development)

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
not implemented. `templates` lists the valid values for `--template`:
`jsonl-documents` and `parquet-documents`.

## Corpus examples

Each block ingests one corpus and then queries it. The question matches that
corpus. `query` searches the newest complete run of the setup, so ingest the
corpus you want to search. A later ingest of the same setup is the one that
gets searched.

### Sample JSONL

Ten local documents: retrieval methods, the first ascent of Everest, filter
coffee, and photosynthesis. File: `data/sample/documents.jsonl`.

```bash
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
uv run raglab query "What is late chunking?" --setup nomic-naive-100-0
```

### NFCorpus

Nutrition articles (BeIR). One document per row, columns `_id`, `text`, `title`.

```bash
uv run raglab ingest \
  https://huggingface.co/datasets/BeIR/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet \
  --template parquet-documents \
  --setup nomic-naive-100-0
uv run raglab query "Does drinking coffee affect health?" --setup nomic-naive-100-0
```

### SciFact

Scientific paper abstracts (BeIR). Same columns and template as NFCorpus.

```bash
uv run raglab ingest \
  https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet \
  --template parquet-documents \
  --setup nomic-naive-100-0
uv run raglab query "How does obesity lead to insulin resistance?" --setup nomic-naive-100-0
```

## Ingestion

```bash
uv run raglab ingest data/sample/documents.jsonl --template jsonl-documents --setup nomic-naive-100-0
```

Needs: `./start.sh`, and the setup's embedding model (see [Setup](#setup)).

Loads the documents with the template, saves them to PostgreSQL, chunks them,
embeds the chunks with the setup's embedding profile and indexes them in Qdrant,
recording the run. Each step is printed. During embedding, a line is printed
after every batch (`Embedded 32/12164 chunks`, and so on). Re-ingesting the
same source does not duplicate anything. Setups whose chunking is not implemented (`late-chunking-v1`)
only store the documents. More templates (`msmarco-passages`, `squad-contexts`,
`markdown-directory`) are planned.

`parquet-documents` reads one document per row from a parquet file. The columns
are `_id`, `text`, and `title`. Other columns are ignored. A blank title is
stored as no title. The source may be a local path or an `http`/`https` URL.
A URL is downloaded into `data/cache/` and reused until it is older than
`CACHE_MAX_AGE_DAYS` (default 30). Delete the cached file to download it again.
The NFCorpus and SciFact commands are in [Corpus examples](#corpus-examples).

## Querying

```bash
uv run raglab query "What is late chunking?" --setup nomic-naive-100-0 --top-k 5
```

Needs: a complete run for the setup (ingest first). Questions for each corpus
are in [Corpus examples](#corpus-examples).

Searches only that setup's newest complete run and prints the `--top-k`
(default 5) best passages with score, document, character range and a text
preview. An earlier corpus ingested with the same setup stays in the collection
and is not searched. Query prefixes (`search_query:`) are not applied yet.

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
