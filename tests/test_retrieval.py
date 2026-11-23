import dataclasses
from collections.abc import Iterator

import pytest

from conftest import FakeEmbedder, unique_id
from rag_lab.chunker import chunk_document
from rag_lab.documents import Document
from rag_lab.indexing import index_chunks
from rag_lab.provenance import ProvenanceStore
from rag_lab.retrieval import search_setup
from rag_lab.settings import get_settings
from rag_lab.setups import Setup, get_setup
from rag_lab.vector_store import QdrantIndex

EMBEDDER = FakeEmbedder(768)
DOCUMENTS = [
    Document("a", " ".join(f"alpha{i}" for i in range(250)), "Doc A", {"k": "a"}),
    Document("b", " ".join(f"beta{i}" for i in range(250)), "Doc B", {"k": "b"}),
]


@pytest.fixture
def store(postgres_settings: None) -> Iterator[ProvenanceStore]:
    settings = get_settings()
    with ProvenanceStore(settings.postgres_dsn, settings.provenance_schema) as store:
        yield store


@pytest.fixture
def index(qdrant_url: str) -> QdrantIndex:
    return QdrantIndex(qdrant_url)


def new_setup() -> Setup:
    return dataclasses.replace(get_setup("nomic-naive-100-0"), id=unique_id("setup"))


def ingest(
    store: ProvenanceStore,
    index: QdrantIndex,
    setup: Setup,
    status: str = "complete",
) -> str:
    corpus = unique_id("corpus")
    store.register_setup(setup)
    run_id = store.start_run(setup.id, "test", corpus)
    store.save_documents(corpus, DOCUMENTS)
    chunks = [c for d in DOCUMENTS for c in chunk_document(d, setup)]
    result = index_chunks(chunks, setup, run_id, corpus, EMBEDDER, index)
    store.add_vector_refs(run_id, result.collection, result.point_ids)
    store.finish_run(run_id, status)
    return run_id


def test_exact_chunk_text_is_the_top_hit_with_postgres_text(
    store: ProvenanceStore, index: QdrantIndex
) -> None:
    setup = new_setup()
    run_id = ingest(store, index, setup)
    chunk = chunk_document(DOCUMENTS[1], setup)[1]

    result = search_setup(chunk.text, setup, EMBEDDER, index, store, top_k=3)

    top = result.passages[0]
    assert result.run_id == run_id and len(result.passages) == 3
    assert (top.rank, top.chunk_id, top.document_id) == (1, chunk.chunk_id, "b")
    assert top.score == pytest.approx(1.0, abs=1e-4)
    assert top.text == chunk.text == DOCUMENTS[1].text[top.start_char : top.end_char]
    assert (top.title, top.metadata) == ("Doc B", {"k": "b"})
    scores = [p.score for p in result.passages]
    assert scores == sorted(scores, reverse=True)


def test_search_is_scoped_to_the_setup(
    store: ProvenanceStore, index: QdrantIndex
) -> None:
    first, second = new_setup(), new_setup()
    ingest(store, index, first)
    ingest(store, index, second)  # same documents, same collection

    result = search_setup("alpha1", first, EMBEDDER, index, store, top_k=50)

    chunk_ids = {c.chunk_id for d in DOCUMENTS for c in chunk_document(d, first)}
    assert {p.chunk_id for p in result.passages} == chunk_ids


def test_only_complete_runs_are_searched(
    store: ProvenanceStore, index: QdrantIndex
) -> None:
    setup = new_setup()
    complete = ingest(store, index, setup)
    failed = store.start_run(setup.id, "test", unique_id("corpus"))
    store.finish_run(failed, "failed")

    result = search_setup("alpha1", setup, EMBEDDER, index, store)

    assert result.run_id == complete


def test_setup_without_a_complete_run_is_rejected(
    store: ProvenanceStore, index: QdrantIndex
) -> None:
    setup = new_setup()
    store.register_setup(setup)

    with pytest.raises(ValueError, match="no complete run"):
        search_setup("q", setup, EMBEDDER, index, store)
