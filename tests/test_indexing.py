from collections.abc import Callable, Sequence

import pytest

from conftest import FakeEmbedder, unique_id
from rag_lab.chunker import chunk_document
from rag_lab.documents import Document
from rag_lab.indexing import index_chunks
from rag_lab.setups import get_setup
from rag_lab.vector_store import QdrantIndex

SETUP = get_setup("nomic-naive-100-0")
DOCUMENT = Document("doc", " ".join(f"w{i}" for i in range(250)))


def test_index_chunks_writes_validated_points(qdrant_url: str) -> None:
    index = QdrantIndex(qdrant_url)
    chunks = chunk_document(DOCUMENT, SETUP)
    run_id = unique_id("run")

    result = index_chunks(chunks, SETUP, run_id, "corpus", FakeEmbedder(768), index)

    assert result.collection == "nomic-embed-text-768"
    assert result.point_ids == [c.chunk_id for c in chunks]
    assert index.count(result.collection, run_id=run_id) == len(chunks) == 3
    assert index.count(result.collection, document_id="doc", run_id=run_id) == 3


def test_payload_contains_chunk_text_and_source_reference(qdrant_url: str) -> None:
    index = QdrantIndex(qdrant_url)
    chunk = chunk_document(DOCUMENT, SETUP)[1]

    result = index_chunks([chunk], SETUP, "run", "corpus", FakeEmbedder(768), index)
    payload = index.payload(result.collection, chunk.chunk_id)

    assert payload["text"] == chunk.text
    assert DOCUMENT.text[payload["start_char"] : payload["end_char"]] == chunk.text
    assert payload["document_id"] == "doc" and payload["corpus_version"] == "corpus"


def test_reindexing_overwrites_points_with_the_new_run(qdrant_url: str) -> None:
    index = QdrantIndex(qdrant_url)
    chunks = chunk_document(DOCUMENT, SETUP)
    first, second = unique_id("run"), unique_id("run")

    index_chunks(chunks, SETUP, first, "corpus", FakeEmbedder(768), index)
    result = index_chunks(chunks, SETUP, second, "corpus", FakeEmbedder(768), index)

    assert index.count(result.collection, run_id=first) == 0
    assert index.count(result.collection, run_id=second) == len(chunks)


def test_dimension_mismatch_with_profile_is_rejected(qdrant_url: str) -> None:
    chunks = chunk_document(DOCUMENT, SETUP)

    with pytest.raises(ValueError, match="returned 8 dimensions, expected 768"):
        index_chunks(chunks, SETUP, "r", "c", FakeEmbedder(8), QdrantIndex(qdrant_url))


def test_inconsistent_vector_lengths_are_rejected(qdrant_url: str) -> None:
    class Uneven:
        def embed(
            self,
            texts: Sequence[str],
            *,
            on_progress: Callable[[int, int], None] | None = None,
        ) -> list[list[float]]:
            return [[1.0] * (i + 1) for i, _ in enumerate(texts)]

    with pytest.raises(ValueError, match="different lengths"):
        index_chunks(
            chunk_document(DOCUMENT, SETUP),
            SETUP,
            "r",
            "c",
            Uneven(),
            QdrantIndex(qdrant_url),
        )


def test_no_chunks_is_rejected(qdrant_url: str) -> None:
    with pytest.raises(ValueError, match="No chunks"):
        index_chunks([], SETUP, "r", "c", FakeEmbedder(768), QdrantIndex(qdrant_url))
