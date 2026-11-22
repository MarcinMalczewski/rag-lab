"""Qdrant access: collections per embedding space, points per chunk."""

import re
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

PAYLOAD_INDEXES = ("setup_id", "run_id", "document_id")
UPSERT_BATCH_SIZE = 128


def collection_name(model: str, dimensions: int) -> str:
    """Name of the collection for one embedding space (model and dimensions)."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"{slug}-{dimensions}"


class QdrantIndex:
    """Thin wrapper over the Qdrant client for the operations the lab needs."""

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self._client = QdrantClient(url=url, api_key=api_key or None)

    def close(self) -> None:
        self._client.close()

    def ensure_collection(self, name: str, dimensions: int) -> bool:
        """Create the cosine collection if missing; return True when created."""
        if self._client.collection_exists(name):
            vectors: Any = self._client.get_collection(name).config.params.vectors
            if vectors.size != dimensions:
                raise ValueError(
                    f"Collection '{name}' has {vectors.size} dimensions, "
                    f"expected {dimensions}."
                )
            return False
        self._client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=dimensions, distance=models.Distance.COSINE
            ),
        )
        for field in PAYLOAD_INDEXES:
            self._client.create_payload_index(
                name, field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        return True

    def upsert(
        self,
        name: str,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """Insert or overwrite points; existing IDs are replaced."""
        for start in range(0, len(ids), UPSERT_BATCH_SIZE):
            end = start + UPSERT_BATCH_SIZE
            self._client.upsert(
                name,
                points=[
                    models.PointStruct(id=i, vector=list(v), payload=p)
                    for i, v, p in zip(
                        ids[start:end],
                        vectors[start:end],
                        payloads[start:end],
                        strict=True,
                    )
                ],
                wait=True,
            )

    def payload(self, name: str, point_id: str) -> dict[str, Any]:
        """Return the payload of one point."""
        (point,) = self._client.retrieve(name, ids=[point_id], with_payload=True)
        return dict(point.payload or {})

    def count(self, name: str, **payload_equals: str) -> int:
        """Exact number of points whose payload fields equal the given values."""
        condition = models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in payload_equals.items()
            ]
        )
        return self._client.count(name, count_filter=condition, exact=True).count
