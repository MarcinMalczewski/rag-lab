from pathlib import Path

import pytest

from rag_lab.cli import build_parser, main
from rag_lab.provenance import ProvenanceStore
from rag_lab.settings import get_settings
from rag_lab.templates import source_checksum

SAMPLE = str(Path(__file__).parents[1] / "data" / "sample" / "documents.jsonl")


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("setups", []),
        ("templates", []),
        (
            "ingest",
            [SAMPLE, "--template", "jsonl-documents", "--setup", "naive-100-0"],
        ),
        ("query", ["What is RAG?", "--setup", "naive-100-0"]),
        ("status", []),
        ("reset", ["--setup", "naive-100-0", "--yes"]),
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
    assert "naive-500-100" in output


def test_templates_command_lists_templates(capsys: object) -> None:
    assert main(["templates"]) == 0
    assert "jsonl-documents" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_ingest_stores_sample_documents(
    postgres_settings: None, capsys: object
) -> None:
    arguments = ["ingest", SAMPLE, "--template", "jsonl-documents", "--setup"]

    assert main([*arguments, "naive-100-0"]) == 0
    assert "Stored 10 documents" in capsys.readouterr().out  # type: ignore[attr-defined]
    assert main([*arguments, "naive-500-100"]) == 0  # same corpus, other setup

    settings = get_settings()
    with ProvenanceStore(settings.postgres_dsn, settings.provenance_schema) as store:
        documents = store.load_documents(source_checksum(Path(SAMPLE)))
    assert len(documents) == 10
