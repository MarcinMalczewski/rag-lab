import dataclasses
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from rag_lab.provenance import ProvenanceStore
from rag_lab.settings import get_settings
from rag_lab.setups import get_setup


@pytest.fixture
def store() -> Iterator[ProvenanceStore]:
    dsn = get_settings().postgres_dsn
    try:
        provenance = ProvenanceStore(dsn, schema=f"test_{uuid.uuid4().hex}")
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not reachable (docker compose up -d postgres)")
    yield provenance
    provenance.drop_schema()
    provenance.close()


def test_run_lifecycle_and_vector_refs(store: ProvenanceStore) -> None:
    setup = get_setup("naive-100-0")
    store.register_setup(setup)
    run_id = store.start_run(setup.id, "jsonl-documents", "v1")
    assert store.run_status(run_id) == "running"
    store.add_vector_refs(run_id, "col", ["b", "a"])
    store.finish_run(run_id)
    assert store.run_status(run_id) == "complete"
    assert store.vector_refs(setup.id) == [("col", "a"), ("col", "b")]


def test_register_setup_is_idempotent_but_immutable(store: ProvenanceStore) -> None:
    setup = get_setup("naive-100-0")
    store.register_setup(setup)
    store.register_setup(setup)
    with pytest.raises(ValueError, match="immutable"):
        store.register_setup(dataclasses.replace(setup, retrieval_mode="hybrid"))


def test_run_requires_registered_setup(store: ProvenanceStore) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.start_run("missing", "t")
