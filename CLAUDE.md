# CLAUDE.md

## Project

`rag-lab` is a lab for reproducible RAG experiments (Python >=3.11, managed with `uv`). It is at an early stage: ingestion, chunking, embedding, Qdrant indexing and dense querying work for the `jsonl-documents` template; evaluation, hybrid search and reranking do not exist yet. `status` and `reset` are placeholders that validate `--setup` and print "not implemented yet"; `query` does dense retrieval; `ingest` loads documents via the template, stores them in PostgreSQL, chunks, embeds (Ollama) and indexes into Qdrant (needs `./start.sh` and Ollama).

## Commands

```bash
uv sync                        # install deps (incl. dev group) into .venv
uv sync --extra hf             # also torch (CPU) + transformers for the huggingface provider
uv run pytest                  # tests (testpaths=tests, pythonpath=src)
uv run pytest tests/test_cli.py::name   # single test
uv run ruff check . && uv run ruff format --check .
uv run mypy                    # strict; covers src and tests
uv run raglab setups           # entry point rag_lab.cli:main
```

## Layout

- `src/rag_lab/cli.py` — argparse CLI. `build_parser()` builds subcommands, each with `set_defaults(handler=...)`; `main(argv)` parses and dispatches, returning an int exit status. Add new commands by adding a subparser plus handler.
- `src/rag_lab/setups.py` — frozen dataclasses `ChunkingProfile`, `EmbeddingProfile` (provider, model, dimensions) and `Setup`. `SETUPS` is generated as embedding profile × chunking profile (`nomic-naive-100-0`, `minilm-naive-500-100`, ...) plus the reserved `late-chunking-v1`. `get_setup()` raises `ValueError` listing valid IDs; `list_setups()` sorts by ID. Add experiments by adding a profile to `CHUNKING_PROFILES` / `EMBEDDING_PROFILES`.
- `src/rag_lab/settings.py` — pydantic-settings `Settings` (reads env / `.env`): Qdrant URL/API key, Postgres, embedding batch size/device, Ollama URL, request timeout/retries. See `.env.example`.
- `src/rag_lab/documents.py` / `templates.py` — normalized `Document` and the ingestion template registry (`IngestionTemplate` protocol with `load(source)`; `TEMPLATES`, `get_template`, `source_checksum`). Only `jsonl-documents` exists; add new templates as classes and register them in `TEMPLATES`. Sample data and questions live in `data/<source>/` (`data/sample/`).
- `src/rag_lab/chunks.py` — `Chunk` model and `make_chunk_id(setup_id, document_id, index)` (UUID5; each chunk has its own ID, `document_id` is shared in the payload).
- `src/rag_lab/chunker.py` — `chunk_document(document, setup)` dispatches on the profile strategy (`_CHUNKERS`); `token_window` is implemented, other strategies raise `NotImplementedError`. Tokens are approximated by a regex (words + punctuation), not the embedding model's tokenizer; chunks carry exact character offsets.
- `src/rag_lab/embeddings.py` — `Embedder` protocol, `OllamaEmbedder` (`/api/embed`, batching, retries), `HuggingFaceEmbedder` (local transformers, mean pooling, L2-normalized; optional extra: `uv sync --extra hf`, a plain `uv sync` removes torch), `get_embedder(settings, profile)` (provider `ollama` or `huggingface` comes from the setup's embedding profile; URL, batch size, device from settings).
- `src/rag_lab/vector_store.py` / `indexing.py` — `QdrantIndex` (collection per embedding space named `<model>-<dims>`, cosine, payload indexes) and `index_chunks()` (embed, upsert, verify count). Payload has `setup_id`, `run_id`, `document_id`, offsets and the chunk `text` (canonical document text is still in PostgreSQL `documents`). qdrant-client is pinned to 1.12.x to match the server image.
- `src/rag_lab/retrieval.py` — `search_setup()`: embeds the query, searches Qdrant filtered by `setup_id` + the setup's latest complete run (`ProvenanceStore.latest_complete_run`), then builds `Passage`s with text/metadata from PostgreSQL (`get_documents` + offsets).
- `src/rag_lab/provenance.py` — `ProvenanceStore` (psycopg, PostgreSQL schema `rag`): setups, runs, documents (keyed by corpus version = source checksum), vector refs. Needs `docker compose up -d` (Qdrant + Postgres 17, host port 5433); DB tests use a disposable Postgres via testcontainers (Docker required), not the dev database.
- `tests/` — pytest tests. `conftest.py` starts a session-wide Postgres 17 testcontainer and a Qdrant one; `postgres_settings` / `qdrant_url` fixtures point settings at them, and `FakeEmbedder` replaces Ollama. Tests share the default schema, so use `unique_id()` for anything they create and assert on.
- `docs/` — design docs (read before implementing features): `architecture.md` (indexing/query flow), `cli.md` (intended command behavior), `plan.md` (milestone order), `rag-strategy.md` / `rag-findings.md` (experiment direction), `sql-schema-analysis.md`.

## Design direction (from docs)

- PostgreSQL = source of truth for documents/provenance; Qdrant = vectors + filterable payloads. Qdrant and Ollama are local dependencies.
- A *setup* fixes chunking, embedding, retrieval and indexing choices; an ingestion *template* (e.g. `squad-contexts`, `msmarco-passages`, `markdown-directory`, `jsonl-documents`) normalizes one input format. Queries and `reset` must be scoped to a single setup.
- Baseline chunking is token-window; hybrid retrieval, reranking and late chunking come later as separate experiments.

## Conventions

- Ruff: line length 88, rules E, F, I, UP, B. mypy strict — fully typed signatures.
- Config via settings/env, no hardcoded secrets.
- Search for existing behavior before adding new logic; reuse/extract helpers rather than duplicating (also for test fixtures). See `.cursor/rules/python-practices.mdc`.
- Tests in `tests/test_cli.py` parametrize over commands; keep them updated when adding commands.
