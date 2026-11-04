# SQL schema analysis for RAG experiments

This is a reference schema for RAG experiments, expressed with a `rag` schema.

## Current model

```text
rag
  └─ note_chunks (ordered passages for note_id + version)
       └─ chunk_embeddings (same passage under many embedding runs)
            └─ embeddings (Qdrant collection + point ID)
                 └─ embedding_runs
                      ├─ embedding_models
                      └─ embedding_methods
```

### Tables and value for experiments

| Table | Current role | RAG value |
|---|---|---|
| `rag.documents` | Document title/content, owner, mutable version, timestamps | SQL source text, ownership filter, update trigger |
| `rag.document_chunks` | Ordered chunk IDs, offsets, preview, hash | Deterministic chunk map and exact passage reconstruction |
| `rag.embedding_models` | Provider/model/revision/dimensions/context/capabilities | Model registry and reproducibility |
| `rag.embedding_methods` | Method name, chunking mode, JSON parameters | Splitter and pooling configuration |
| `rag.embedding_runs` | Model + method + code/parameters/status/times | Experiment and migration boundary |
| `rag.embeddings` | Embedding ID, run, Qdrant collection/point ID | Vector-store provenance and safe cleanup |
| `rag.chunk_embeddings` | Passage-to-vector join per run | Side-by-side model/method comparisons |

The composite primary keys and unique constraints on `note_chunks` and
`chunk_embeddings` are good protections against duplicate chunk positions and
duplicate links within a run. The separate join table correctly allows one
passage to have many experimental embeddings.

## Important gaps

### 1. Document versions are not immutable or relationally enforced

`rag.documents` has `id BIGSERIAL PRIMARY KEY` and a mutable `version` column.
`document_chunks` stores `(document_id, version, ...)`, but its foreign key only checks
`document_id` against `rag.documents(id)`; it does not prove that the referenced
`version` is an actual stored document version. There is no primary key or unique
constraint on `(id, version)`, no version history table, and no active-version
mapping.

This prevents reliable historical retrieval and makes the proposed atomic
cutover/rollback protocol impossible with the current table alone. For the
first simple experiment, pin the current document row and treat `version` as a
test-controlled generation number. Before testing edits or concurrent searches,
add immutable version rows, for example:

```text
documents (document_id, owner_id, title, created_at)
document_versions (document_id, version, content, content_hash, created_at, is_active)
document_chunks (document_id, version, ...)
```

Then make `document_chunks(document_id, version)` reference `document_versions` and enforce
one active version per document with a partial unique index or transaction-safe
application logic.

### 2. Chunk text is only a preview

`text_preview` is appropriate for Qdrant payload/debugging, but it is not the
canonical chunk text. Exact reconstruction depends on `documents.content` and on a
documented canonicalization policy. Record the canonicalization version in the
embedding method/run parameters; otherwise offsets can become invalid after
whitespace or normalization changes.

### 3. Run status and integrity are application-enforced

`status` is free-form text, and the schema does not require a completed run to
have a consistent set of chunk links or require a Qdrant point to exist. The
indexer should define allowed states (`queued`, `running`, `completed`,
`failed`, `superseded`) and use an outbox/job record or reconciliation pass for
SQL/Qdrant failures.

### 4. Missing fields for comparable retrieval experiments

The strategy proposes, but the reference schema does not yet include, several useful
fields: distance metric, tokenizer, pooling strategy, dataset/corpus version,
embedding profile ID, evaluation records, and modality/source references for
images. These can start in JSONB, but fields used for filtering and joins should
eventually become typed columns.

### 5. No explicit ownership on chunks or vectors

Ownership is available through the document join, but Qdrant payloads must carry
`owner_id` to filter before exposing results. The SQL schema should ensure the
indexing worker derives owner from the verified document row rather than accepting
it from client input. Add tests for cross-owner leakage.

### 6. No vector in PostgreSQL by design

The `embeddings` table stores Qdrant identifiers, not vector values. That is a
reasonable separation for this architecture, but experiments must either run
Qdrant or provide a test double. SQL alone cannot reproduce similarity search.

## Recommended minimal additions

For a practical RAG test harness, prioritize these changes in order:

1. Add a real immutable `note_versions` table and make chunk FKs reference it.
2. Add a typed or JSONB `embedding_profile`/distance metric and record the
   canonicalization version, separators, size, and overlap in every run.
3. Add an `embedding_evaluations` table containing benchmark identity, metrics,
   and run annotations.
4. Add an indexing job/outbox state so SQL commit, Qdrant upsert, active-version
   cutover, retry, and cleanup can be tested separately.
5. Add indexes for expected queries, such as chunks by `(note_id, version,
   chunk_index)` (already present), links by `(run_id, note_id, version)`, and
   runs by status.

## Suggested experiment queries

These are logical query shapes for the harness, not migrations:

```sql
-- Fetch ordered source passages for a selected version.
SELECT chunk_id, chunk_index, start_offset, end_offset, content_hash
FROM rag.document_chunks
WHERE document_id = :document_id AND version = :version
ORDER BY chunk_index;

-- Select only vectors from one reproducible embedding run.
SELECT ce.note_id, ce.version, ce.chunk_id, e.qdrant_collection, e.qdrant_id
FROM rag.chunk_embeddings AS ce
JOIN rag.embeddings AS e ON e.embedding_id = ce.embedding_id
WHERE ce.run_id = :run_id;
```

The vector similarity query belongs in Qdrant and must include the selected
profile/run and owner/version constraints. Use the SQL result only to resolve
canonical content and provenance.

