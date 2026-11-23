"""Setup-scoped dense retrieval: Qdrant finds chunks, PostgreSQL supplies text."""

from dataclasses import dataclass
from typing import Any

from rag_lab.embeddings import Embedder
from rag_lab.provenance import ProvenanceStore
from rag_lab.setups import Setup
from rag_lab.vector_store import QdrantIndex, collection_name


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk with canonical text and metadata from PostgreSQL."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    title: str | None
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """Passages plus what was searched, for display and evaluation."""

    run_id: str
    corpus_version: str
    collection: str
    passages: list[Passage]


def search_setup(
    query: str,
    setup: Setup,
    embedder: Embedder,
    index: QdrantIndex,
    store: ProvenanceStore,
    top_k: int = 5,
) -> SearchResult:
    """Embed the query and return the best passages of the setup's latest
    complete run, with text and metadata read from PostgreSQL."""
    run = store.latest_complete_run(setup.id)
    if run is None:
        raise ValueError(
            f"Setup '{setup.id}' has no complete run; run `raglab ingest` first."
        )
    run_id, corpus_version = run
    (vector,) = embedder.embed([query])
    collection = collection_name(setup.embedding.model, len(vector))
    hits = index.search(collection, vector, top_k, setup_id=setup.id, run_id=run_id)
    documents = store.get_documents(
        corpus_version, sorted({str(hit.payload["document_id"]) for hit in hits})
    )
    passages = []
    for rank, hit in enumerate(hits, start=1):
        document = documents[str(hit.payload["document_id"])]
        start, end = int(hit.payload["start_char"]), int(hit.payload["end_char"])
        passages.append(
            Passage(
                rank=rank,
                score=hit.score,
                chunk_id=str(hit.payload["chunk_id"]),
                document_id=document.document_id,
                title=document.title,
                text=document.text[start:end],
                start_char=start,
                end_char=end,
                metadata=document.metadata,
            )
        )
    return SearchResult(run_id, corpus_version, collection, passages)
