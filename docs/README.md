# RAG lab documentation

This folder records the RAG design, experiment strategy, and implementation
decisions for this project. Strategy documents describe proposed direction and
may change as experiments produce evidence.

## Documents

- [RAG findings and experiment guidance](rag-findings.md) — vector-store,
  chunking, embedding, retrieval, and evaluation recommendations.
- [RAG strategy working draft](rag-strategy.md) — provisional experiment and
  Qdrant-collection management direction for this project.
- [Architecture](architecture.md) — indexing and query flow.
- [CLI](cli.md) — initial commands for predefined RAG setups.
- [Implementation plan](plan.md) — short milestone sequence.
- [SQL schema analysis](sql-schema-analysis.md) — current tables, useful
  relationships, gaps, and a migration path for RAG experiments.

## Current direction

- PostgreSQL is the source of truth for canonical documents, ownership, and
  experiment provenance.
- Qdrant stores searchable vectors and filterable retrieval payloads.
- The initial chunking baseline is independent recursive or token-window
  chunks; alternative strategies are evaluated as explicit setups.
- Embedding profiles, owner filtering, incremental updates, cross-modal search,
  and profile migrations are planned experiment dimensions.
- Qdrant and PostgreSQL 17 run locally via Docker Compose (`./start.sh`; Qdrant
  on port 6333, PostgreSQL on 5433). Ollama is an intended local dependency.
- `rag_lab.provenance` stores setups, runs, and vector references in the
  PostgreSQL `rag` schema, together with the normalized documents saved by
  `raglab ingest`. Tables are created on first use; versioned migrations are
  not yet in place.
- `raglab ingest` embeds chunks with Ollama and indexes them in Qdrant; search
  (`raglab query`) remains to be implemented and evaluated in this project.

