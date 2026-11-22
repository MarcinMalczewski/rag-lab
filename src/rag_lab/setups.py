"""Predefined RAG experiment setups: a chunking profile plus an embedding profile."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingProfile:
    """Chunking choices used to create document passages."""

    id: str
    strategy: str
    chunk_size_tokens: int | None
    chunk_overlap_tokens: int | None


@dataclass(frozen=True)
class EmbeddingProfile:
    """Everything that defines one embedding space."""

    id: str
    provider: str
    model: str
    dimensions: int | None


@dataclass(frozen=True)
class Setup:
    """A named, reproducible RAG configuration."""

    id: str
    description: str
    chunking: ChunkingProfile
    embedding: EmbeddingProfile
    retrieval_mode: str


CHUNKING_PROFILES: tuple[ChunkingProfile, ...] = (
    ChunkingProfile("naive-100-0", "token_window", 100, 0),
    ChunkingProfile("naive-500-100", "token_window", 500, 100),
)

EMBEDDING_PROFILES: tuple[EmbeddingProfile, ...] = (
    EmbeddingProfile("nomic", "ollama", "nomic-embed-text", 768),
    EmbeddingProfile(
        "minilm", "huggingface", "sentence-transformers/all-MiniLM-L6-v2", 384
    ),
)

# Reserved until a late-chunking implementation chooses its model.
_LATE_CHUNKING = Setup(
    id="late-chunking-v1",
    description="Reserved profile for the first late-chunking experiment.",
    chunking=ChunkingProfile("late-chunking", "late_chunking", None, None),
    embedding=EmbeddingProfile("model-dependent", "none", "model-dependent", None),
    retrieval_mode="dense",
)


def _combine(chunking: ChunkingProfile, embedding: EmbeddingProfile) -> Setup:
    return Setup(
        id=f"{embedding.id}-{chunking.id}",
        description=(
            f"{chunking.chunk_size_tokens}-token windows, "
            f"{chunking.chunk_overlap_tokens} overlap, {embedding.model} "
            f"({embedding.provider}), dense retrieval."
        ),
        chunking=chunking,
        embedding=embedding,
        retrieval_mode="dense",
    )


SETUPS: dict[str, Setup] = {
    setup.id: setup
    for setup in (
        *(
            _combine(chunking, embedding)
            for embedding in EMBEDDING_PROFILES
            for chunking in CHUNKING_PROFILES
        ),
        _LATE_CHUNKING,
    )
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
