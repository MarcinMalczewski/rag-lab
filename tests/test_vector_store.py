import pytest

from conftest import unique_id
from rag_lab.vector_store import QdrantIndex, collection_name


def test_collection_name_slugs_model_and_adds_dimensions() -> None:
    assert collection_name("nomic-embed-text", 768) == "nomic-embed-text-768"
    assert collection_name("Org/Model:v1.5", 8) == "org-model-v1-5-8"


def test_ensure_collection_creates_once_and_checks_dimensions(
    qdrant_url: str,
) -> None:
    index = QdrantIndex(qdrant_url)
    name = unique_id("col")

    assert index.ensure_collection(name, 4) is True
    assert index.ensure_collection(name, 4) is False
    with pytest.raises(ValueError, match="has 4 dimensions, expected 8"):
        index.ensure_collection(name, 8)


def test_upsert_overwrites_and_count_filters_by_payload(qdrant_url: str) -> None:
    index = QdrantIndex(qdrant_url)
    name = unique_id("col")
    index.ensure_collection(name, 2)
    ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]

    index.upsert(name, ids, [[1, 0], [0, 1]], [{"setup_id": "a"}, {"setup_id": "b"}])
    index.upsert(name, ids[:1], [[1, 0]], [{"setup_id": "b"}])  # same ID: replaced

    assert index.count(name, setup_id="a") == 0
    assert index.count(name, setup_id="b") == 2
