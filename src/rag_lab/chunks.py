"""Chunk model and deterministic chunk identities."""

import uuid
from dataclasses import dataclass

CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "rag-lab.chunks")


def make_chunk_id(setup_id: str, document_id: str, index: int) -> str:
    """Return a stable UUID for one chunk of a document under one setup.

    The same inputs always give the same ID, so re-ingesting overwrites the
    existing Qdrant point. Different chunks, documents or setups never collide.
    UUIDs are valid Qdrant point IDs.
    """
    name = "\x1f".join((setup_id, document_id, str(index)))
    return str(uuid.uuid5(CHUNK_NAMESPACE, name))


@dataclass(frozen=True)
class Chunk:
    """A passage of a document with a stable source reference."""

    chunk_id: str
    document_id: str
    index: int
    text: str
    start_char: int
    end_char: int
