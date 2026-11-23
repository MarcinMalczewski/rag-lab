from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import FakeEmbedder
from rag_lab.cli import build_parser, main
from rag_lab.embeddings import EmbeddingError
from rag_lab.provenance import ProvenanceStore
from rag_lab.settings import get_settings
from rag_lab.templates import source_checksum
from rag_lab.vector_store import QdrantIndex

SAMPLE = str(Path(__file__).parents[1] / "data" / "sample" / "documents.jsonl")


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("setups", []),
        ("templates", []),
        (
            "ingest",
            [SAMPLE, "--template", "jsonl-documents", "--setup", "nomic-naive-100-0"],
        ),
        ("query", ["What is RAG?", "--setup", "nomic-naive-100-0"]),
        ("status", []),
        ("reset", ["--setup", "nomic-naive-100-0", "--yes"]),
    ],
)
def test_allowed_commands_are_accepted(command: str, arguments: list[str]) -> None:
    parsed = build_parser().parse_args([command, *arguments])

    assert parsed.command == command


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(["unknown-command"])


def test_setups_command_lists_predefined_setups(capsys: object) -> None:
    assert main(["setups"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "nomic-naive-500-100" in output


def test_templates_command_lists_templates(capsys: object) -> None:
    assert main(["templates"]) == 0
    assert "jsonl-documents" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_ingest_stores_chunks_and_indexes_vectors(
    postgres_settings: None,
    qdrant_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: object,
) -> None:
    monkeypatch.setattr("rag_lab.cli.get_embedder", lambda *_: FakeEmbedder(768))
    arguments = ["ingest", SAMPLE, "--template", "jsonl-documents", "--setup"]

    assert main([*arguments, "nomic-naive-100-0"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Stored 10 documents" in output
    assert "Chunked into" in output
    assert "Run finished with status complete." in output
    assert main([*arguments, "nomic-naive-100-0"]) == 0  # re-ingest overwrites points
    assert main([*arguments, "nomic-naive-500-100"]) == 0  # same corpus, other setup

    settings = get_settings()
    with ProvenanceStore(settings.postgres_dsn, settings.provenance_schema) as store:
        documents = store.load_documents(source_checksum(Path(SAMPLE)))
        refs = store.vector_refs("nomic-naive-100-0")
    assert len(documents) == 10
    index = QdrantIndex(qdrant_url)
    assert index.count("nomic-embed-text-768", setup_id="nomic-naive-100-0") == len(
        refs
    )
    assert index.count("nomic-embed-text-768", setup_id="nomic-naive-500-100") == 10


def test_ingest_reports_embedding_failure(
    postgres_settings: None,
    qdrant_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: object,
) -> None:
    class Failing:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise EmbeddingError("model missing")

    monkeypatch.setattr("rag_lab.cli.get_embedder", lambda *_: Failing())

    status = main(
        [
            "ingest",
            SAMPLE,
            "--template",
            "jsonl-documents",
            "--setup",
            "nomic-naive-500-100",
        ]
    )

    assert status == 1
    assert "run marked failed: model missing" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_ingest_skips_embedding_for_unimplemented_chunking(
    postgres_settings: None, capsys: object
) -> None:
    status = main(
        [
            "ingest",
            SAMPLE,
            "--template",
            "jsonl-documents",
            "--setup",
            "late-chunking-v1",
        ]
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert status == 0
    assert "Chunking skipped" in output
    assert "documents_stored" in output


def test_query_returns_passages_after_ingest(
    postgres_settings: None,
    qdrant_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: object,
) -> None:
    monkeypatch.setattr("rag_lab.cli.get_embedder", lambda *_: FakeEmbedder(768))
    ingest = ["ingest", SAMPLE, "--template", "jsonl-documents", "--setup"]
    assert main([*ingest, "nomic-naive-100-0"]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]

    assert (
        main(["query", "late chunking", "--setup", "nomic-naive-100-0", "--top-k", "3"])
        == 0
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "1. score" in output and "3. score" in output and "4. score" not in output
    assert "doc-" in output


def test_query_reports_setup_without_complete_run(
    postgres_settings: None,
    qdrant_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: object,
) -> None:
    monkeypatch.setattr("rag_lab.cli.get_embedder", lambda *_: FakeEmbedder(384))

    assert main(["query", "anything", "--setup", "minilm-naive-500-100"]) == 1
    assert "no complete run" in capsys.readouterr().out  # type: ignore[attr-defined]
