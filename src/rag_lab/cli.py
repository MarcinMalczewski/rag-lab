"""Command-line shell for RAG lab experiments."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from rag_lab.chunker import chunk_document
from rag_lab.chunks import Chunk
from rag_lab.documents import Document
from rag_lab.embeddings import EmbeddingError, get_embedder
from rag_lab.indexing import index_chunks
from rag_lab.provenance import ProvenanceStore
from rag_lab.retrieval import search_setup
from rag_lab.settings import Settings, get_settings
from rag_lab.setups import Setup, get_setup, list_setups
from rag_lab.templates import get_template, list_templates, source_checksum
from rag_lab.vector_store import QdrantIndex

PREVIEW_CHARS = 300


def _print_setups(_: argparse.Namespace) -> int:
    for setup in list_setups():
        print(f"{setup.id}: {setup.description}")
    return 0


def _print_templates(_: argparse.Namespace) -> int:
    for template in list_templates():
        print(f"{template.id}: {template.description}")
    return 0


def _chunk_all(documents: list[Document], setup: Setup) -> dict[str, list[Chunk]]:
    return {d.document_id: chunk_document(d, setup) for d in documents}


def _ingest(args: argparse.Namespace) -> int:
    setup = get_setup(args.setup)
    template = get_template(args.template)
    source = Path(args.source)
    chunking = setup.chunking
    print(
        f"Setup:    {setup.id} ({chunking.strategy}, "
        f"size {chunking.chunk_size_tokens}, overlap {chunking.chunk_overlap_tokens}, "
        f"embedding {setup.embedding.id}: {setup.embedding.provider}/"
        f"{setup.embedding.model}, {setup.embedding.dimensions} dims, "
        f"{setup.retrieval_mode})"
    )
    print(f"Template: {template.id}")
    print(f"Source:   {source}")

    documents = template.load(source)
    chunks_by_document: dict[str, list[Chunk]] | None
    chunk_note = ""
    try:
        chunks_by_document = _chunk_all(documents, setup)
    except NotImplementedError as error:
        chunks_by_document = None
        chunk_note = f"Chunking skipped: {error}"
    words = sum(len(document.text.split()) for document in documents)
    print(f"Loaded {len(documents)} documents ({words} words):")
    for document in documents:
        chunks = (
            f", {len(chunks_by_document[document.document_id])} chunks"
            if chunks_by_document
            else ""
        )
        print(
            f"  {document.document_id}  {document.title or '-'} "
            f"({len(document.text.split())} words{chunks})"
        )
    corpus_version = source_checksum(source)
    print(f"Corpus version (source checksum): {corpus_version[:12]}")
    all_chunks = (
        [c for chunks in chunks_by_document.values() for c in chunks]
        if chunks_by_document
        else []
    )
    if chunks_by_document is not None:
        print(f"Chunked into {len(all_chunks)} chunks (in memory).")
    else:
        print(chunk_note)

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
            print(
                f"Stored {len(documents)} documents "
                f"({len(documents) - existing} new, {existing} already stored)."
            )
            if not all_chunks:
                store.finish_run(run_id, "documents_stored")
                print("Run finished with status documents_stored.")
                return 0
            try:
                _index(setup, run_id, corpus_version, all_chunks, store, settings)
            except Exception as error:
                store.finish_run(run_id, "failed")
                print(f"Indexing failed, run marked failed: {error}")
                return 1
            store.finish_run(run_id)
            print("Run finished with status complete.")
    except psycopg.OperationalError as error:
        print(f"Cannot reach PostgreSQL (is ./start.sh running?): {error}")
        return 1
    return 0


def _index(
    setup: Setup,
    run_id: str,
    corpus_version: str,
    chunks: list[Chunk],
    store: ProvenanceStore,
    settings: Settings,
) -> None:
    profile = setup.embedding
    embedder = get_embedder(settings, profile)
    print(
        f"Embedding {len(chunks)} chunks with {profile.provider}/{profile.model} "
        f"(batch size {settings.embedding_batch_size})..."
    )
    index = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    try:
        result = index_chunks(
            chunks,
            setup,
            run_id,
            corpus_version,
            embedder,
            index,
        )
    finally:
        index.close()
    state = "created" if result.created_collection else "existing"
    print(
        f"Qdrant collection {result.collection} ({state}, "
        f"{result.dimensions} dimensions)."
    )
    store.add_vector_refs(run_id, result.collection, result.point_ids)
    print(
        f"Indexed {len(result.point_ids)} points; verified in Qdrant and "
        "recorded in PostgreSQL."
    )


def _query(args: argparse.Namespace) -> int:
    setup = get_setup(args.setup)
    settings = get_settings()
    print(f"Setup: {setup.id} (embedding {setup.embedding.id})")
    print(f"Query: {args.query!r} (top {args.top_k})")
    try:
        with ProvenanceStore(
            settings.postgres_dsn, settings.provenance_schema
        ) as store:
            embedder = get_embedder(settings, setup.embedding)
            index = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
            try:
                result = search_setup(
                    args.query, setup, embedder, index, store, args.top_k
                )
            finally:
                index.close()
    except psycopg.OperationalError as error:
        print(f"Cannot reach PostgreSQL (is ./start.sh running?): {error}")
        return 1
    except (ValueError, EmbeddingError) as error:
        print(f"Query failed: {error}")
        return 1
    print(
        f"Searched run {result.run_id} (corpus {result.corpus_version[:12]}) "
        f"in collection {result.collection}."
    )
    if not result.passages:
        print("No results.")
    for passage in result.passages:
        preview = " ".join(passage.text.split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        print(
            f"\n{passage.rank}. score {passage.score:.4f}  {passage.document_id}  "
            f"{passage.title or '-'}  [chars {passage.start_char}-{passage.end_char}]"
        )
        print(f"   {preview}")
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
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.set_defaults(handler=_query)

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
