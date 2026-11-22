"""Chunkers that split documents into passages according to a setup."""

import re
from collections.abc import Callable

from rag_lab.chunks import Chunk, make_chunk_id
from rag_lab.documents import Document
from rag_lab.setups import Setup

# Approximate tokens: a word (letters, digits, underscore) or one punctuation mark.
# Offline and deterministic; it is not the embedding model's own tokenizer.
_TOKEN = re.compile(r"\w+|[^\w\s]")


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character offsets of every token in the text."""
    return [(match.start(), match.end()) for match in _TOKEN.finditer(text)]


def token_window_chunks(document: Document, setup: Setup) -> list[Chunk]:
    """Split a document into fixed-size token windows with optional overlap."""
    profile = setup.chunking
    size = profile.chunk_size_tokens
    overlap = profile.chunk_overlap_tokens
    if size is None or overlap is None or size <= 0 or not 0 <= overlap < size:
        raise ValueError(
            f"Setup '{setup.id}': token_window needs chunk_size_tokens > 0 and "
            f"0 <= chunk_overlap_tokens < chunk_size_tokens "
            f"(got {size}, {overlap})."
        )
    spans = token_spans(document.text)
    chunks: list[Chunk] = []
    start = 0
    while start < len(spans):
        end = min(start + size, len(spans))
        first, last = spans[start][0], spans[end - 1][1]
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(setup.id, document.document_id, len(chunks)),
                document_id=document.document_id,
                index=len(chunks),
                text=document.text[first:last],
                start_char=first,
                end_char=last,
            )
        )
        if end == len(spans):
            break
        start += size - overlap
    return chunks


_CHUNKERS: dict[str, Callable[[Document, Setup], list[Chunk]]] = {
    "token_window": token_window_chunks,
}


def chunk_document(document: Document, setup: Setup) -> list[Chunk]:
    """Chunk a document with the strategy of the setup's chunking profile."""
    try:
        chunker = _CHUNKERS[setup.chunking.strategy]
    except KeyError as error:
        raise NotImplementedError(
            f"No chunker for strategy '{setup.chunking.strategy}' (setup '{setup.id}')."
        ) from error
    return chunker(document, setup)
