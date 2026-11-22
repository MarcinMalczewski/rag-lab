# Implementation plan

1. [x] Create the Python project structure, configuration registry, and CLI shell.
2. [x] Run Qdrant locally (`docker compose up -d`) and add a minimal provenance
   store (PostgreSQL, `rag_lab.provenance`).
3. Implement ingestion templates:
   1. [x] Template registry and normalized `Document` model.
   2. [x] `jsonl-documents` template (manual data, `data/sample/`).
   3. [ ] MS MARCO template.
   4. [ ] SQuAD template.
4. Add normalized document storage and deterministic chunk identities:
   1. [x] Store normalized documents in PostgreSQL, keyed by corpus version
      (source checksum).
   2. [x] Deterministic chunk IDs (`rag_lab.chunks`) and `Chunk` model.
5. [x] Implement baseline token-window chunking profiles
   (`rag_lab.chunker`; tokens are approximated by words and punctuation).
6. Add embedding providers and index document chunks in Qdrant.
7. Implement setup-scoped query and dense retrieval, fetching canonical
   passage text and metadata from PostgreSQL by the returned IDs.
8. Add retrieval evaluation using dataset relevance labels.
9. Compare chunking profiles and record experiment results.
10. Add hybrid retrieval, reranking, and late chunking as separate experiments.

