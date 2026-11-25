"""Code-defined ingestion templates that normalize input formats to documents."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from rag_lab.documents import Document


class IngestionTemplate(Protocol):
    """Adapter that converts one input format into normalized documents."""

    @property
    def id(self) -> str:
        """Stable template ID used in `--template`."""
        ...

    @property
    def description(self) -> str:
        """One-line summary for CLI output."""
        ...

    def load(self, source: Path) -> list[Document]:
        """Parse a source into documents or raise ValueError with the location."""
        ...


@dataclass(frozen=True)
class JsonlDocumentsTemplate:
    """One document per JSONL row, with configurable field names."""

    id: str = "jsonl-documents"
    description: str = "One document per JSONL row (id, text, title, metadata)."
    id_field: str = "id"
    text_field: str = "text"
    title_field: str = "title"
    metadata_field: str = "metadata"

    def load(self, source: Path) -> list[Document]:
        documents: list[Document] = []
        seen: set[str] = set()
        with source.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, start=1):
                if line.strip():
                    document = self._parse(line, f"{source}:{number}")
                    if document.document_id in seen:
                        raise ValueError(
                            f"{source}:{number}: duplicate id '{document.document_id}'"
                        )
                    seen.add(document.document_id)
                    documents.append(document)
        if not documents:
            raise ValueError(f"{source}: no documents found")
        return documents

    def _parse(self, line: str, location: str) -> Document:
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON ({error.msg})") from error
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a JSON object")
        document_id = self._required_text(row, self.id_field, location)
        text = self._required_text(row, self.text_field, location)
        title = row.get(self.title_field)
        metadata = row.get(self.metadata_field, {})
        if title is not None and not isinstance(title, str):
            raise ValueError(f"{location}: '{self.title_field}' must be a string")
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}: '{self.metadata_field}' must be an object")
        return Document(document_id, text, title, metadata)

    @staticmethod
    def _required_text(row: dict[str, Any], name: str, location: str) -> str:
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location}: '{name}' must be a non-empty string")
        return value


@dataclass(frozen=True)
class ParquetDocumentsTemplate:
    """One document per parquet row, with configurable column names."""

    id: str = "parquet-documents"
    description: str = "One document per parquet row (_id, text, title)."
    id_field: str = "_id"
    text_field: str = "text"
    title_field: str = "title"

    def load(self, source: Path) -> list[Document]:
        """Load one document per row from a parquet file.

        Expects ``id_field``, ``text_field``, and ``title_field``. Other columns
        are ignored. A null or blank title is stored as None. Raises ValueError
        for a missing column, a blank or non-string id or text, a non-string
        title, a duplicate id, or a table with no rows. Row numbers start at 1.

        >>> ParquetDocumentsTemplate().title_field
        'title'
        """
        table = pq.read_table(source)
        columns = set(table.column_names)
        for name in (self.id_field, self.text_field, self.title_field):
            if name not in columns:
                raise ValueError(f"{source}: missing column '{name}'")
        rows = table.select(
            [self.id_field, self.text_field, self.title_field]
        ).to_pylist()
        documents: list[Document] = []
        seen: set[str] = set()
        for number, row in enumerate(rows, start=1):
            location = f"{source}:{number}"
            document_id = JsonlDocumentsTemplate._required_text(
                row, self.id_field, location
            )
            text = JsonlDocumentsTemplate._required_text(row, self.text_field, location)
            title = self._title(row, location)
            if document_id in seen:
                raise ValueError(f"{location}: duplicate id '{document_id}'")
            seen.add(document_id)
            documents.append(Document(document_id, text, title, {}))
        if not documents:
            raise ValueError(f"{source}: no documents found")
        return documents

    def _title(self, row: dict[str, Any], location: str) -> str | None:
        """Return the title, or None when it is null or blank.

        >>> ParquetDocumentsTemplate()._title({"title": "  "}, "row") is None
        True
        """
        value = row.get(self.title_field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{location}: '{self.title_field}' must be a string")
        if not value.strip():
            return None
        return value


TEMPLATES: dict[str, IngestionTemplate] = {
    template.id: template
    for template in (JsonlDocumentsTemplate(), ParquetDocumentsTemplate())
}


def get_template(template_id: str) -> IngestionTemplate:
    """Return a template or raise a clear error for an unknown ID."""
    try:
        return TEMPLATES[template_id]
    except KeyError as error:
        available = ", ".join(sorted(TEMPLATES))
        raise ValueError(
            f"Unknown template '{template_id}'. Available: {available}"
        ) from error


def list_templates() -> tuple[IngestionTemplate, ...]:
    """Return templates in a stable order for CLI output and tests."""
    return tuple(TEMPLATES[template_id] for template_id in sorted(TEMPLATES))


def _source_files(source: Path) -> Iterator[Path]:
    return iter([source]) if source.is_file() else iter(sorted(source.rglob("*")))


def source_checksum(source: Path) -> str:
    """SHA-256 of a file, or of all files in a directory (paths and contents)."""
    digest = hashlib.sha256()
    for path in _source_files(source):
        if path.is_file():
            if source.is_dir():
                digest.update(path.relative_to(source).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
