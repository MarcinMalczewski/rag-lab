"""Predefined RAG experiment setups."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingProfile:
    """Chunking choices used to create document passages."""

    strategy: str
    chunk_size_tokens: int | None
    chunk_overlap_tokens: int | None


@dataclass(frozen=True)
class Setup:
    """A named, reproducible RAG configuration."""

    id: str
    description: str
    chunking: ChunkingProfile
    embedding_model: str
    retrieval_mode: str


SETUPS: dict[str, Setup] = {
    "naive-100-0": Setup(
        id="naive-100-0",
        description="100-token windows with no overlap and dense retrieval.",
        chunking=ChunkingProfile("token_window", 100, 0),
        embedding_model="nomic-embed-text",
        retrieval_mode="dense",
    ),
    "naive-500-100": Setup(
        id="naive-500-100",
        description="500-token windows with 100-token overlap and dense retrieval.",
        chunking=ChunkingProfile("token_window", 500, 100),
        embedding_model="nomic-embed-text",
        retrieval_mode="dense",
    ),
    "late-chunking-v1": Setup(
        id="late-chunking-v1",
        description="Reserved profile for the first late-chunking experiment.",
        chunking=ChunkingProfile("late_chunking", None, None),
        embedding_model="model-dependent",
        retrieval_mode="dense",
    ),
}


def get_setup(setup_id: str) -> Setup:
    """Return a setup or raise a clear error for an unknown ID."""
    try:
        return SETUPS[setup_id]
    except KeyError as error:
        available = ", ".join(sorted(SETUPS))
        raise ValueError(
            f"Unknown setup '{setup_id}'. Available: {available}"
        ) from error


def list_setups() -> tuple[Setup, ...]:
    """Return setups in a stable order for CLI output and tests."""
    return tuple(SETUPS[setup_id] for setup_id in sorted(SETUPS))
