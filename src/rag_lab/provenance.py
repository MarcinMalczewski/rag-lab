"""Minimal PostgreSQL provenance store for setups, runs, and vector references."""

from dataclasses import asdict
from typing import Self

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from rag_lab.documents import Document
from rag_lab.setups import Setup

_TABLES = """
CREATE TABLE IF NOT EXISTS {schema}.setups (
    setup_id TEXT PRIMARY KEY,
    definition JSONB NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS {schema}.documents (
    corpus_version TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}',
    PRIMARY KEY (corpus_version, document_id)
);
CREATE TABLE IF NOT EXISTS {schema}.runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setup_id TEXT NOT NULL REFERENCES {schema}.setups(setup_id),
    template_id TEXT NOT NULL,
    corpus_version TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS {schema}.vector_refs (
    run_id UUID NOT NULL REFERENCES {schema}.runs(run_id),
    collection TEXT NOT NULL,
    point_id TEXT NOT NULL,
    PRIMARY KEY (collection, point_id)
);
"""


class ProvenanceStore:
    """Records which setups, runs, and Qdrant points exist."""

    def __init__(self, dsn: str, schema: str = "rag") -> None:
        self._schema = sql.Identifier(schema)
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(self._schema)
        )
        self._conn.execute(sql.SQL(_TABLES).format(schema=self._schema))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _q(self, query: str) -> sql.Composed:
        return sql.SQL(query).format(schema=self._schema)

    def register_setup(self, setup: Setup) -> None:
        """Store a setup definition; reject a changed definition for the same ID."""
        definition = asdict(setup)
        self._conn.execute(
            self._q(
                "INSERT INTO {schema}.setups (setup_id, definition) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING"
            ),
            (setup.id, Jsonb(definition)),
        )
        row = self._conn.execute(
            self._q("SELECT definition FROM {schema}.setups WHERE setup_id = %s"),
            (setup.id,),
        ).fetchone()
        if row is None or row[0] != definition:
            raise ValueError(f"Setup '{setup.id}' is immutable and has changed.")

    def save_documents(self, corpus_version: str, documents: list[Document]) -> None:
        """Store normalized documents; a corpus version is immutable, so re-saving
        the same version leaves existing rows untouched."""
        with self._conn.cursor() as cursor:
            cursor.executemany(
                self._q(
                    "INSERT INTO {schema}.documents "
                    "(corpus_version, document_id, title, text, metadata) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
                ),
                [
                    (
                        corpus_version,
                        document.document_id,
                        document.title,
                        document.text,
                        Jsonb(document.metadata),
                    )
                    for document in documents
                ],
            )

    def count_documents(self, corpus_version: str) -> int:
        row = self._conn.execute(
            self._q(
                "SELECT count(*) FROM {schema}.documents WHERE corpus_version = %s"
            ),
            (corpus_version,),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def load_documents(self, corpus_version: str) -> list[Document]:
        """Return the documents of a corpus version ordered by document ID."""
        rows = self._conn.execute(
            self._q(
                "SELECT document_id, text, title, metadata FROM {schema}.documents "
                "WHERE corpus_version = %s ORDER BY document_id"
            ),
            (corpus_version,),
        ).fetchall()
        return [Document(str(i), str(t), title, meta) for i, t, title, meta in rows]

    def start_run(
        self, setup_id: str, template_id: str, corpus_version: str | None = None
    ) -> str:
        """Create a running run for a registered setup and return its ID."""
        row = self._conn.execute(
            self._q(
                "INSERT INTO {schema}.runs (setup_id, template_id, corpus_version) "
                "VALUES (%s, %s, %s) RETURNING run_id"
            ),
            (setup_id, template_id, corpus_version),
        ).fetchone()
        assert row is not None
        return str(row[0])

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        self._conn.execute(
            self._q(
                "UPDATE {schema}.runs SET status = %s, finished_at = now() "
                "WHERE run_id = %s"
            ),
            (status, run_id),
        )

    def run_status(self, run_id: str) -> str:
        row = self._conn.execute(
            self._q("SELECT status FROM {schema}.runs WHERE run_id = %s"),
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown run '{run_id}'.")
        return str(row[0])

    def add_vector_refs(
        self, run_id: str, collection: str, point_ids: list[str]
    ) -> None:
        with self._conn.cursor() as cursor:
            cursor.executemany(
                self._q("INSERT INTO {schema}.vector_refs VALUES (%s, %s, %s)"),
                [(run_id, collection, point_id) for point_id in point_ids],
            )

    def vector_refs(self, setup_id: str) -> list[tuple[str, str]]:
        """Return (collection, point_id) pairs for a setup, e.g. for reset."""
        rows = self._conn.execute(
            self._q(
                "SELECT v.collection, v.point_id FROM {schema}.vector_refs v "
                "JOIN {schema}.runs r ON r.run_id = v.run_id "
                "WHERE r.setup_id = %s ORDER BY v.collection, v.point_id"
            ),
            (setup_id,),
        ).fetchall()
        return [(str(c), str(p)) for c, p in rows]
