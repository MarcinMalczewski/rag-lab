"""Command-line shell for RAG lab experiments."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from rag_lab.chunker import chunk_document
from rag_lab.provenance import ProvenanceStore
from rag_lab.settings import get_settings
from rag_lab.setups import get_setup, list_setups
from rag_lab.templates import get_template, list_templates, source_checksum


def _print_setups(_: argparse.Namespace) -> int:
    for setup in list_setups():
        print(f"{setup.id}: {setup.description}")
    return 0


def _print_templates(_: argparse.Namespace) -> int:
    for template in list_templates():
        print(f"{template.id}: {template.description}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    setup = get_setup(args.setup)
    template = get_template(args.template)
    source = Path(args.source)
    chunking = setup.chunking
    print(
        f"Setup:    {setup.id} ({chunking.strategy}, "
        f"size {chunking.chunk_size_tokens}, overlap {chunking.chunk_overlap_tokens}, "
        f"{setup.embedding_model}, {setup.retrieval_mode})"
    )
    print(f"Template: {template.id}")
    print(f"Source:   {source}")

    documents = template.load(source)
    words = sum(len(document.text.split()) for document in documents)
    chunk_counts: dict[str, int] | None
    try:
        chunk_counts = {
            document.document_id: len(chunk_document(document, setup))
            for document in documents
        }
    except NotImplementedError as error:
        chunk_counts = None
        chunk_note = f"Chunking skipped: {error}"
    print(f"Loaded {len(documents)} documents ({words} words):")
    for document in documents:
        chunks = (
            f", {chunk_counts[document.document_id]} chunks" if chunk_counts else ""
        )
        print(
            f"  {document.document_id}  {document.title or '-'} "
            f"({len(document.text.split())} words{chunks})"
        )
    if chunk_counts is not None:
        print(f"Chunked into {sum(chunk_counts.values())} chunks (in memory).")
    else:
        print(chunk_note)
    corpus_version = source_checksum(source)
    print(f"Corpus version (source checksum): {corpus_version[:12]}")

    settings = get_settings()
    print(
        f"Connecting to PostgreSQL {settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db} (schema {settings.provenance_schema})..."
    )
    try:
        with ProvenanceStore(
            settings.postgres_dsn, settings.provenance_schema
        ) as store:
            store.register_setup(setup)
            print(f"Registered setup {setup.id}.")
            run_id = store.start_run(setup.id, template.id, corpus_version)
            print(f"Started run {run_id}.")
            existing = store.count_documents(corpus_version)
            store.save_documents(corpus_version, documents)
            store.finish_run(run_id, "documents_stored")
    except psycopg.OperationalError as error:
        print(f"Cannot reach PostgreSQL (is ./start.sh running?): {error}")
        return 1
    print(
        f"Stored {len(documents)} documents "
        f"({len(documents) - existing} new, {existing} already stored)."
    )
    print("Run finished with status documents_stored.")
    print("Next: embedding and indexing are not implemented yet.")
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

    templates_parser = commands.add_parser(
        "templates", help="List ingestion templates."
    )
    templates_parser.set_defaults(handler=_print_templates)

    ingest_parser = commands.add_parser("ingest", help="Ingest source data.")
    ingest_parser.add_argument("source")
    ingest_parser.add_argument("--template", required=True)
    ingest_parser.add_argument("--setup", required=True)
    ingest_parser.set_defaults(handler=_ingest)

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
