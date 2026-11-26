"""Embed chunks and write them to Qdrant."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rag_lab.chunks import Chunk
from rag_lab.embeddings import Embedder
from rag_lab.setups import Setup
from rag_lab.vector_store import QdrantIndex, collection_name


@dataclass(frozen=True)
class IndexResult:
    """Where the chunks went and what was verified."""

    collection: str
    created_collection: bool
    dimensions: int
    point_ids: list[str]


def chunk_payload(
    setup: Setup, run_id: str, corpus_version: str, chunk: Chunk
) -> dict[str, Any]:
    """Filterable metadata stored next to each vector."""
    return {
        "setup_id": setup.id,
        "run_id": run_id,
        "corpus_version": corpus_version,
        "chunking_profile": setup.chunking.strategy,
        "embedding_profile": setup.embedding.id,
        "embedding_model": setup.embedding.model,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.index,
        "text": chunk.text,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def index_chunks(
    chunks: list[Chunk],
    setup: Setup,
    run_id: str,
    corpus_version: str,
    embedder: Embedder,
    index: QdrantIndex,
    on_progress: Callable[[int, int], None] | None = None,
) -> IndexResult:
    """Embed and upsert chunks, then verify the stored point count for the run.

    When on_progress is set, it is called as on_progress(done, total) after
    each embedding batch.
    """
    if not chunks:
        raise ValueError("No chunks to index.")
    vectors = embedder.embed([chunk.text for chunk in chunks], on_progress=on_progress)
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedder returned vectors of different lengths.")
    expected = setup.embedding.dimensions
    if expected is not None and dimensions != expected:
        raise ValueError(
            f"Model '{setup.embedding.model}' returned {dimensions} dimensions, "
            f"expected {expected} (embedding profile '{setup.embedding.id}')."
        )
    name = collection_name(setup.embedding.model, dimensions)
    created = index.ensure_collection(name, dimensions)
    ids = [chunk.chunk_id for chunk in chunks]
    index.upsert(
        name,
        ids,
        vectors,
        [chunk_payload(setup, run_id, corpus_version, chunk) for chunk in chunks],
    )
    stored = index.count(name, setup_id=setup.id, run_id=run_id)
    if stored != len(ids):
        raise ValueError(
            f"Validation failed: {stored} points stored for run, expected {len(ids)}."
        )
    return IndexResult(name, created, dimensions, ids)
