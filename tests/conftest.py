import random
import uuid
from collections.abc import Iterator, Sequence

import pytest
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.qdrant import QdrantContainer


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


@pytest.fixture(scope="session")
def qdrant() -> Iterator[QdrantContainer]:
    """One disposable Qdrant container (same version as docker-compose.yml)."""
    with QdrantContainer("qdrant/qdrant:v1.12.4") as container:
        yield container


@pytest.fixture
def qdrant_url(qdrant: QdrantContainer, monkeypatch: pytest.MonkeyPatch) -> str:
    """URL of the test Qdrant, also exported as QDRANT_URL for `get_settings()`."""
    url = f"http://{qdrant.get_container_host_ip()}:{qdrant.get_exposed_port(6333)}"
    monkeypatch.setenv("QDRANT_URL", url)
    return url


class FakeEmbedder:
    """Deterministic embedder for tests; no model required."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            rng = random.Random(text)
            vectors.append([rng.random() for _ in range(self.dimensions)])
        return vectors
