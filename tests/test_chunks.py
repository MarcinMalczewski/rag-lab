import uuid

from rag_lab.chunks import make_chunk_id


def test_chunk_id_is_deterministic_uuid() -> None:
    first = make_chunk_id("nomic-naive-100-0", "doc-001", 3)

    assert first == make_chunk_id("nomic-naive-100-0", "doc-001", 3)
    assert str(uuid.UUID(first)) == first


def test_chunk_ids_differ_by_chunk_document_and_setup() -> None:
    ids = {
        make_chunk_id("nomic-naive-100-0", "doc-001", 0),
        make_chunk_id("nomic-naive-100-0", "doc-001", 1),
        make_chunk_id("nomic-naive-100-0", "doc-002", 0),
        make_chunk_id("nomic-naive-500-100", "doc-001", 0),
    }

    assert len(ids) == 4


def test_chunk_id_parts_are_not_ambiguous() -> None:
    assert make_chunk_id("a", "b1", 2) != make_chunk_id("a", "b", 12)
