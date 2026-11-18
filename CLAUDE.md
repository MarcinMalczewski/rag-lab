# CLAUDE.md

## Project

`rag-lab` is a lab for reproducible RAG experiments (Python >=3.11, managed with `uv`). It is at an early stage: only the CLI shell, settings, and setup registry exist. Ingestion, chunking, embedding, Qdrant indexing, and querying are **not implemented yet**; `ingest`, `query`, `status`, and `reset` are placeholders that validate `--setup` and print "not implemented yet"; `ingest` currently only loads and validates documents via the chosen template.

## Commands

```bash
uv sync                        # install deps (incl. dev group) into .venv
uv run pytest                  # tests (testpaths=tests, pythonpath=src)
uv run pytest tests/test_cli.py::name   # single test
uv run ruff check . && uv run ruff format --check .
uv run mypy                    # strict; covers src and tests
uv run raglab setups           # entry point rag_lab.cli:main
```

## Layout

- `src/rag_lab/cli.py` — argparse CLI. `build_parser()` builds subcommands, each with `set_defaults(handler=...)`; `main(argv)` parses and dispatches, returning an int exit status. Add new commands by adding a subparser plus handler.
- `src/rag_lab/setups.py` — frozen dataclasses `Setup` / `ChunkingProfile` and the `SETUPS` registry (`naive-100-0`, `naive-500-100`, `late-chunking-v1`). `get_setup()` raises `ValueError` listing valid IDs; `list_setups()` returns setups sorted by ID. New experiments = new registry entries (setups are code-defined, not config files).
- `src/rag_lab/settings.py` — pydantic-settings `Settings` (reads env / `.env`): Qdrant URL/API key, embedding provider/model/dimensions/batch size, Ollama URL, request timeout/retries. See `.env.example`.
- `src/rag_lab/documents.py` / `templates.py` — normalized `Document` and the ingestion template registry (`IngestionTemplate` protocol with `load(source)`; `TEMPLATES`, `get_template`, `source_checksum`). Only `jsonl-documents` exists; add new templates as classes and register them in `TEMPLATES`. Sample data and questions live in `data/<source>/` (`data/sample/`).
- `src/rag_lab/provenance.py` — `ProvenanceStore` (psycopg, PostgreSQL schema `rag`): setups, runs, vector refs. Needs `docker compose up -d` (Qdrant + Postgres 17, host port 5433); DB tests skip if Postgres is unreachable.
- `tests/` — pytest tests for CLI and setups.
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
