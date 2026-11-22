import dataclasses
from collections.abc import Iterator

import psycopg
import pytest

from conftest import unique_id
from rag_lab.documents import Document
from rag_lab.provenance import ProvenanceStore
from rag_lab.settings import get_settings
from rag_lab.setups import Setup, get_setup


@pytest.fixture
def store(postgres_settings: None) -> Iterator[ProvenanceStore]:
    settings = get_settings()
    with ProvenanceStore(settings.postgres_dsn, settings.provenance_schema) as store:
        yield store


def unique_setup() -> Setup:
    return dataclasses.replace(get_setup("naive-100-0"), id=unique_id("setup"))


def test_run_lifecycle_and_vector_refs(store: ProvenanceStore) -> None:
    setup = unique_setup()
    store.register_setup(setup)
    run_id = store.start_run(setup.id, "jsonl-documents", "v1")
    assert store.run_status(run_id) == "running"
    store.add_vector_refs(run_id, "col", [unique_id("b"), unique_id("a")])
    store.finish_run(run_id)
    assert store.run_status(run_id) == "complete"
    assert len(store.vector_refs(setup.id)) == 2


def test_register_setup_is_idempotent_but_immutable(store: ProvenanceStore) -> None:
    setup = unique_setup()
    store.register_setup(setup)
    store.register_setup(setup)
    with pytest.raises(ValueError, match="immutable"):
        store.register_setup(dataclasses.replace(setup, retrieval_mode="hybrid"))


def test_run_requires_registered_setup(store: ProvenanceStore) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.start_run(unique_id("missing"), "t")


def test_documents_round_trip_and_are_immutable_per_corpus(
    store: ProvenanceStore,
) -> None:
    v1, v2 = unique_id("v1"), unique_id("v2")
    documents = [
        Document("b", "text b", None, {}),
        Document("a", "text a", "Title", {"topic": "x"}),
    ]
    store.save_documents(v1, documents)
    store.save_documents(v1, [Document("a", "changed", None, {})])
    store.save_documents(v2, [Document("a", "other corpus", None, {})])

    assert store.load_documents(v1) == [documents[1], documents[0]]
    assert [d.text for d in store.load_documents(v2)] == ["other corpus"]
    assert store.load_documents(unique_id("missing")) == []
