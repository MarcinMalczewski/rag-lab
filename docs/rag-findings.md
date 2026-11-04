# RAG findings and experiment guidance

## Executive summary

The current direction is a hybrid RAG architecture:

```text
canonical document text + ownership + versions  -> PostgreSQL
chunk/vector lookup + filterable payload    -> Qdrant
embedding generation                        -> Ollama initially
answer generation                           -> future provider abstraction
```

The most useful baseline for `rag-lab` is independent per-chunk embedding with
deterministic chunk metadata. Start with this before comparing late chunking,
structure-aware splitting, multimodal retrieval, or rerankers.

Important status distinction: this is a strategy for the RAG lab, not a
complete working pipeline. Each component should be implemented and evaluated
independently before it becomes a baseline.

## Baseline indexing contract

1. Canonicalize the source text using a documented, versioned normalization
   policy.
2. Split it with `RecursiveCharacterTextSplitter` using explicit
   `chunk_size`, `chunk_overlap`, and separators.
3. Persist ordered chunks with character offsets, a stable `chunk_id`, and a
   SHA-256-style `content_hash`.
4. Embed each chunk independently. Batch requests for throughput, but do not
   concatenate unrelated chunks into one embedding input.
5. Store vectors in a Qdrant collection/namespace dedicated to one embedding
   profile. Payload should include `document_id`, owner, document version, chunk ID,
   offsets, modality, and enough preview/debug metadata to inspect a hit.
6. Persist SQL provenance linking the chunk to the model, method, run, and
   Qdrant point ID.

The embedding profile must include at least provider, exact model/revision,
dimensions, distance metric, and chunking/method configuration. Vectors from
different profiles must not be compared in one similarity search.

## Chunking strategy

The recommended first strategy is recursive character splitting. It attempts to
preserve paragraphs and natural separators while enforcing the model context
limit. Keep overlap modest: it helps boundary recall but increases vector count,
storage, and re-embedding work.

Required invariants for experiments:

- A chunk must fit the embedding model's actual token limit; never silently
  truncate a chunk or the entire document.
- Offsets must refer to the exact canonical representation used for indexing.
- Chunk identity, content hash, and embedding-run identity are separate
  concepts. A content hash alone is not enough to reuse a vector.
- Edits near an overlap boundary can change neighboring chunks; re-index every
  affected range, not only the visibly edited chunk.

Late chunking, contextual windows, structure-aware splitting, and token-level
representations are explicitly deferred in the source plan. They are good
follow-up experiments against the recursive-split baseline.

## Retrieval flow

For a text query:

1. Embed the query with the model's query/retrieval instruction or mode.
2. Search only the active embedding profile in Qdrant.
3. Apply owner/authorization and version filters in the vector search.
4. Fetch canonical document text from PostgreSQL using the returned IDs and version.
5. Reconstruct exact passages from offsets, optionally merge adjacent hits, and
   add controlled surrounding context before answer generation.

Authorization is a correctness requirement, not merely an optimization. The
plans call for owner isolation tests in both SQL and Qdrant.

## Embedding/provider observations

`nomic-embed-text` through Ollama with 768 dimensions is a reasonable initial
baseline. An embedding provider should return one vector per input in input
order, reject empty batches and inconsistent dimensions, and retry transient
failures.

The provider abstraction deliberately keeps embedding calls separate from LLM
chat/generation and from prompt orchestration. Keep that separation in test
experiments so retrieval quality is measurable independently of answer quality.

Multilingual and multimodal candidates may use a shared-space model such as
Jina CLIP v2, or separate text/image models. Do not mix such vectors unless the
models were trained for a shared space. For image search, route text and image
queries through a cross-modal profile, or use separate indexes and merge at the
document level.

## Incremental updates and lifecycle

The intended update protocol is: save a new document version, generate its chunk
map, reuse unchanged chunk embeddings, insert new vectors, atomically switch the
active version, and lazily garbage-collect superseded vectors after a grace
period. Keep the old profile during embedding-model migrations and cut over only
after the new collection is complete.

This protocol should be an explicit experiment in `rag-lab`, because it tests
more than top-k quality: determinism, cache reuse, stale-result safety,
rollback, partial failure recovery, and cleanup.

## Recommended experiment matrix

Use a fixed multilingual corpus and labelled query set. Record retrieval recall
and precision, answer faithfulness/quality if an LLM is used, latency, token or
inference cost, vector count, and storage size.

| Stage | Variable | Baseline | Comparisons |
|---|---|---|---|
| 1 | chunking | recursive character split | size/overlap grid |
| 2 | embedding | Ollama `nomic-embed-text` | other exact model profiles |
| 3 | retrieval | vector top-k | metadata filters, adjacent-hit merge, reranking |
| 4 | lifecycle | full indexing | hash reuse, edits, orphan cleanup |
| 5 | modality | text only | image extraction and cross-modal merge |
| 6 | generation | retrieved passages | context windows and answer evaluation |

Do not compare runs with different corpus, query labels, profile, or filtering
rules without recording those differences. Store evaluation results separately
from vectors so one run can be benchmarked repeatedly.

## Risks and corrections found in the source material

- Use `Qdrant` consistently in code and documentation.
- The reference schema has provenance for Qdrant points but does not store the
  vector itself or a dedicated evaluation table.
- The current reference document table does not provide immutable version rows; see
  [SQL schema analysis](sql-schema-analysis.md).

