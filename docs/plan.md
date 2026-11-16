# Implementation plan

1. Create the Python project structure, configuration registry, and CLI shell.
2. Run Qdrant locally and add a minimal provenance store.
3. Implement ingestion templates for manual data, MS MARCO, and SQuAD.
4. Add normalized document storage and deterministic chunk identities.
5. Implement baseline token-window chunking profiles.
6. Add embedding providers and index document chunks in Qdrant.
7. Implement setup-scoped query and dense retrieval.
8. Add retrieval evaluation using dataset relevance labels.
9. Compare chunking profiles and record experiment results.
10. Add hybrid retrieval, reranking, and late chunking as separate experiments.

