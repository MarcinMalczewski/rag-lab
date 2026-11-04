# RAG strategy (working draft)

This document is a starting point for `rag-lab`, not an agreed architecture.
It turns the ideas in [RAG findings](rag-findings.md) into experiments that can
be compared and revised.

## Direction for now

- **Vector database:** Qdrant is the vector database of choice.
- **Goal:** compare RAG configurations reproducibly rather than prematurely
  choosing one chunking or retrieval approach.
- **Principle:** chunking, embedding, indexing, retrieval, and ranking settings
  must be independently configurable and recorded with every result.

## Ingestion templates

Input parsing is independent from chunking. An ingestion template is a named,
code-defined adapter that converts one input format into a common internal
document representation:

```text
document_id, text, optional title, metadata
```

Initial template IDs should include:

| Template ID | Intended input | Normalization details |
|---|---|---|
| `msmarco-passages` | MS MARCO passage/document files | Retain source IDs and relevance labels for evaluation. |
| `squad-contexts` | SQuAD JSON | Index unique contexts; retain questions and answer spans as evaluation data. |
| `markdown-directory` | Manually maintained Markdown files | Use file path/front matter as source metadata. |
| `jsonl-documents` | One document per JSONL row | Map configured ID, text, title, and metadata fields. |

The ingestion run must record the template ID, source location, source checksum
or dataset release, and normalized corpus version. This keeps comparisons fair:
two chunking setups should not appear different merely because their source data
was parsed differently.

## Chunking experiments

The indexing pipeline should select a chunker by a named profile rather than
hard-coding a splitting algorithm. A profile declares the strategy and every
parameter that affects its output.

Initial profiles to implement and test:

| Profile ID | Strategy | Chunk size | Overlap | Details |
|---|---|---:|---:|---|
| `naive-100-0` | token/window | 100 tokens | 0 tokens | Minimal baseline; likely weak at boundaries. |
| `naive-500-100` | token/window | 500 tokens | 100 tokens | Larger contextual baseline with overlap. |
| `late-chunking-*` | late chunking | model-dependent | model-dependent | Deferred until its model-specific implementation and evaluation contract are chosen. |

The naive profiles are examples, not fixed defaults. Add other profile IDs for
different window sizes, overlap sizes, separators, or document-aware splitting.
Every chunker must return ordered chunks, exact offsets or another stable source
reference, and deterministic chunk identifiers for the same document and
profile.

Late chunking should be represented as a separate strategy, not as a flag on
the naive splitter. Its profile must record the embedding model/revision and
the method parameters needed to reproduce its contextual embeddings.

## Configuration as an experiment setup

An **experiment setup** is the complete retrieval configuration. It should have
a stable setup ID and an immutable stored definition. At minimum, record:

| Area | Record |
|---|---|
| Corpus | ingestion template, source/corpus version, canonicalization version, document filters |
| Chunking | strategy, profile ID, size, overlap, separators, late-chunking parameters |
| Embeddings | provider, model, revision, dimensions, query instruction/mode, distance metric |
| Index | Qdrant collection, payload schema, HNSW/quantization settings if changed |
| Retrieval | top-k, metadata/owner filters, hybrid-search method and weights |
| Ranking | reranker model/configuration and adjacent-hit or document-level merge policy |
| Generation | optional LLM/context-packing/prompt version, kept separate from retrieval metrics |
| Evaluation | query/relevance-set version, metrics, timestamps, code revision, status |

Example setup IDs: `nomic-naive-500-100-dense`,
`nomic-naive-500-100-hybrid-bm25`, or `model-x-late-chunking-dense`. IDs are
for people; the full immutable configuration is the source of truth.

## Managing experiments in Qdrant

There are two useful isolation boundaries:

1. **Collection per compatible embedding space.** A collection should contain
   vectors that can safely be compared: same model/revision, dimensions,
   distance metric, and normally the same modality. This prevents invalid
   mixed-model search.
2. **Payload/run metadata for experiments within that space.** Each point
   should include at least `setup_id`, `run_id`, `corpus_version`,
   `chunking_profile`, `document_id`, `document_version`, `chunk_id`, and any
   authorization fields. Search filters select one completed setup/run.

This is the proposed default because different chunk sizes or overlaps can use
the same embedding space and do not inherently need separate collections. It
reduces collection sprawl while keeping experiments isolated through filters.

A dedicated collection per setup is still a valid option when the setup changes
index-level settings, requires destructive rebuilds, has very different scale
or retention requirements, or needs strong operational isolation. It is likely
the simplest early implementation, but can become expensive and cumbersome as
the setup matrix grows.

### Proposed lifecycle

1. Register the immutable setup and create or choose its compatible Qdrant
   collection.
2. Create a run with corpus and code versions.
3. Index chunks with `setup_id` and `run_id` payload fields.
4. Validate vector count, dimensions, and payload completeness.
5. Mark the run complete, then permit it in evaluation/search filters.
6. Keep results and metrics linked to the setup/run; clean up failed or expired
   runs according to an explicit retention policy.

Use a relational experiment/provenance store alongside Qdrant for setup
definitions, runs, metrics, and vector point references. Qdrant should serve
similarity search, not become the only source of experiment history.

## Hybrid search

Hybrid search is an experiment dimension, not an automatic default. Candidate
profiles include dense-only retrieval, lexical-only retrieval, and dense plus
BM25/sparse retrieval with a documented fusion algorithm and weight. Record
the sparse model/index, top-k values, fusion method, and weights in the setup.

## Decisions still open

- Which chunking profiles form the first evaluation matrix?
- Which embedding models and query instructions should be the baseline?
- Should the first implementation use one collection per setup for simplicity,
  then consolidate compatible setups, or start directly with filtered runs?
- Which hybrid-search implementation and fusion method should be tested?
- What corpus and labelled query set will make the comparisons meaningful?
- What retrieval and answer-quality metrics define a successful setup?

