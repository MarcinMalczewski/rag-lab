import uuid
from collections.abc import Iterator

import pytest
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    """One disposable PostgreSQL 17 container for the whole test session."""
    with PostgresContainer("postgres:17", driver=None) as container:
        yield container


@pytest.fixture
def postgres_settings(
    postgres: PostgresContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point `get_settings()` at the test container, never the dev database.

    Tests share the container's default schema, so each one must use unique IDs
    (see `unique_id`) for anything it creates and asserts on.
    """
    monkeypatch.setenv("POSTGRES_HOST", postgres.get_container_host_ip())
    monkeypatch.setenv("POSTGRES_PORT", str(postgres.get_exposed_port(5432)))
    monkeypatch.setenv("POSTGRES_USER", postgres.username)
    monkeypatch.setenv("POSTGRES_PASSWORD", postgres.password)
    monkeypatch.setenv("POSTGRES_DB", postgres.dbname)


def unique_id(prefix: str) -> str:
    """Return an ID no other test will use."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
