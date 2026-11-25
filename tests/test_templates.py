from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag_lab.templates import (
    JsonlDocumentsTemplate,
    ParquetDocumentsTemplate,
    get_template,
    list_templates,
    source_checksum,
)

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "documents.jsonl"


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "docs.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_registry_lists_and_resolves_templates() -> None:
    assert [template.id for template in list_templates()] == [
        "jsonl-documents",
        "parquet-documents",
    ]
    assert get_template("jsonl-documents").id == "jsonl-documents"
    with pytest.raises(ValueError, match="Unknown template 'x'"):
        get_template("x")


def test_sample_documents_load() -> None:
    documents = get_template("jsonl-documents").load(SAMPLE)

    assert len(documents) == 10
    assert len({document.document_id for document in documents}) == 10
    assert all(document.text and document.title for document in documents)
    assert documents[0].metadata["topic"] == "technical"


def test_optional_fields_and_blank_lines(tmp_path: Path) -> None:
    path = write(tmp_path, '{"id": "a", "text": "hello"}\n\n')

    (document,) = JsonlDocumentsTemplate().load(path)

    assert (document.title, document.metadata) == (None, {})


def test_configured_field_names(tmp_path: Path) -> None:
    path = write(tmp_path, '{"pid": "p1", "body": "hi", "name": "T"}\n')
    template = JsonlDocumentsTemplate(
        id_field="pid", text_field="body", title_field="name"
    )

    (document,) = template.load(path)

    assert (document.document_id, document.text, document.title) == ("p1", "hi", "T")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", r":1: invalid JSON"),
        ('{"id": "a", "text": "x"}\n[1]\n', r":2: expected a JSON object"),
        ('{"text": "x"}\n', r":1: 'id' must be a non-empty string"),
        ('{"id": "a", "text": " "}\n', r":1: 'text' must be a non-empty string"),
        ('{"id": "a", "text": "x", "title": 3}\n', r"'title' must be a string"),
        ('{"id": "a", "text": "x", "metadata": []}\n', r"'metadata' must be an object"),
        ('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n', r":2: duplicate id"),
        ("\n", r"no documents found"),
    ],
)
def test_invalid_input_reports_location(
    tmp_path: Path, content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        JsonlDocumentsTemplate().load(write(tmp_path, content))


def write_parquet(
    path: Path, rows: list[dict[str, object]], schema: pa.Schema | None = None
) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def test_parquet_rows_become_documents(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "corpus.parquet",
        [
            {"_id": "a", "text": "hello", "title": "Hello", "url": "http://a"},
            {"_id": "b", "text": "world", "title": "World", "url": "http://b"},
        ],
    )

    documents = ParquetDocumentsTemplate().load(path)

    assert [(d.document_id, d.text, d.title, d.metadata) for d in documents] == [
        ("a", "hello", "Hello", {}),
        ("b", "world", "World", {}),
    ]


def test_blank_parquet_title_is_missing(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "corpus.parquet",
        [
            {"_id": "a", "text": "hello", "title": ""},
            {"_id": "b", "text": "world", "title": None},
        ],
    )

    documents = ParquetDocumentsTemplate().load(path)

    assert [document.title for document in documents] == [None, None]


@pytest.mark.parametrize(
    ("rows", "schema", "message"),
    [
        (
            [
                {"_id": "a", "text": "x", "title": "t"},
                {"_id": "a", "text": "y", "title": "u"},
            ],
            None,
            r":2: duplicate id",
        ),
        (
            [{"_id": "a", "text": "x"}],
            pa.schema([("_id", pa.string()), ("text", pa.string())]),
            r"missing column 'title'",
        ),
        (
            [],
            pa.schema(
                [
                    ("_id", pa.string()),
                    ("text", pa.string()),
                    ("title", pa.string()),
                ]
            ),
            r"no documents found",
        ),
        ([{"_id": 1, "text": "x", "title": "t"}], None, r":1:.*_id"),
        ([{"_id": "a", "text": "x", "title": 3}], None, r":1:.*title"),
        ([{"_id": "a", "text": "  ", "title": "t"}], None, r"text"),
    ],
)
def test_invalid_parquet_reports_location(
    tmp_path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema | None,
    message: str,
) -> None:
    path = write_parquet(tmp_path / "corpus.parquet", rows, schema)

    with pytest.raises(ValueError, match=message):
        ParquetDocumentsTemplate().load(path)


def test_source_checksum_tracks_content(tmp_path: Path) -> None:
    path = write(tmp_path, "a")
    first = source_checksum(path)
    path.write_text("b", encoding="utf-8")

    assert source_checksum(path) != first
    assert source_checksum(tmp_path) == source_checksum(tmp_path)
