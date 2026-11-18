"""Normalized document representation shared by all ingestion templates."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document after template normalization."""

    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
