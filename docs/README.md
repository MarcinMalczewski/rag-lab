# RAG lab documentation

This folder records the RAG design, experiment strategy, and implementation
decisions for this project. Strategy documents describe proposed direction and
may change as experiments produce evidence.

## Documents

- [RAG findings and experiment guidance](rag-findings.md) — vector-store,
  chunking, embedding, retrieval, and evaluation recommendations.
- [RAG strategy working draft](rag-strategy.md) — provisional experiment and
  Qdrant-collection management direction for this project.
- [CLI](cli.md) — initial commands for predefined RAG setups.
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
- Qdrant and Ollama are intended local dependencies; vector indexing and search
  integration remain to be implemented and evaluated in this project.

