"""Command-line shell for RAG lab experiments."""

import argparse
from collections.abc import Callable, Sequence

from rag_lab.setups import get_setup, list_setups


def _print_setups(_: argparse.Namespace) -> int:
    for setup in list_setups():
        print(f"{setup.id}: {setup.description}")
    return 0


def _not_implemented(args: argparse.Namespace) -> int:
    if hasattr(args, "setup"):
        get_setup(args.setup)
    print(f"{args.command} is not implemented yet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing commands."""
    parser = argparse.ArgumentParser(prog="raglab")
    commands = parser.add_subparsers(dest="command", required=True)

    setups_parser = commands.add_parser("setups", help="List predefined setups.")
    setups_parser.set_defaults(handler=_print_setups)

    ingest_parser = commands.add_parser("ingest", help="Ingest source data.")
    ingest_parser.add_argument("source")
    ingest_parser.add_argument("--template", required=True)
    ingest_parser.add_argument("--setup", required=True)
    ingest_parser.set_defaults(handler=_not_implemented)

    query_parser = commands.add_parser("query", help="Query an indexed setup.")
    query_parser.add_argument("query")
    query_parser.add_argument("--setup", required=True)
    query_parser.set_defaults(handler=_not_implemented)

    status_parser = commands.add_parser("status", help="Show index status.")
    status_parser.set_defaults(handler=_not_implemented)

    reset_parser = commands.add_parser("reset", help="Reset one indexed setup.")
    reset_parser.add_argument("--setup", required=True)
    reset_parser.add_argument("--yes", action="store_true")
    reset_parser.set_defaults(handler=_not_implemented)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return its exit status."""
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
