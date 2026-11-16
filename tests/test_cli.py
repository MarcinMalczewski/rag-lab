import pytest

from rag_lab.cli import build_parser, main


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("setups", []),
        (
            "ingest",
            ["data", "--template", "jsonl-documents", "--setup", "naive-100-0"],
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


def test_ingest_shell_validates_setup(capsys: object) -> None:
    assert (
        main(
            [
                "ingest",
                "data",
                "--template",
                "jsonl-documents",
                "--setup",
                "naive-100-0",
            ]
        )
        == 0
    )
    assert "not implemented yet" in capsys.readouterr().out  # type: ignore[attr-defined]
