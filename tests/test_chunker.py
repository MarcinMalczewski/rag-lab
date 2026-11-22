import dataclasses
from pathlib import Path

import pytest

from rag_lab.chunker import chunk_document, token_spans
from rag_lab.chunks import make_chunk_id
from rag_lab.documents import Document
from rag_lab.setups import ChunkingProfile, Setup, get_setup
from rag_lab.templates import get_template

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "documents.jsonl"


def setup_with(size: int | None, overlap: int | None) -> Setup:
    return dataclasses.replace(
        get_setup("naive-100-0"),
        id="test-setup",
        chunking=ChunkingProfile("token_window", size, overlap),
    )


def words(count: int) -> Document:
    return Document("doc", " ".join(f"w{i}" for i in range(count)))


def test_token_spans_count_words_and_punctuation() -> None:
    text = "Hello, world!"

    assert [text[a:b] for a, b in token_spans(text)] == ["Hello", ",", "world", "!"]


def test_windows_without_overlap() -> None:
    chunks = chunk_document(words(250), setup_with(100, 0))

    assert [len(token_spans(c.text)) for c in chunks] == [100, 100, 50]
    assert chunks[0].text.startswith("w0 ") and chunks[1].text.startswith("w100 ")


def test_windows_with_overlap_share_tokens() -> None:
    chunks = chunk_document(words(250), setup_with(100, 20))

    assert [len(token_spans(c.text)) for c in chunks] == [100, 100, 90]
    assert chunks[1].text.startswith("w80 ")
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_short_document_is_one_chunk() -> None:
    document = words(5)
    (chunk,) = chunk_document(document, setup_with(100, 20))

    assert chunk.text == document.text
    assert (chunk.start_char, chunk.end_char) == (0, len(document.text))


def test_offsets_match_source_and_ids_are_deterministic() -> None:
    document = words(250)
    setup = setup_with(100, 20)
    chunks = chunk_document(document, setup)

    assert chunks == chunk_document(document, setup)
    assert [c.index for c in chunks] == [0, 1, 2]
    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.chunk_id == make_chunk_id(setup.id, "doc", chunk.index)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("size", "overlap"), [(0, 0), (100, 100), (100, -1), (None, 0)]
)
def test_invalid_window_parameters_are_rejected(
    size: int | None, overlap: int | None
) -> None:
    with pytest.raises(ValueError, match="token_window needs"):
        chunk_document(words(10), setup_with(size, overlap))


def test_unimplemented_strategy_is_reported() -> None:
    with pytest.raises(NotImplementedError, match="late_chunking"):
        chunk_document(words(10), get_setup("late-chunking-v1"))


@pytest.mark.parametrize("setup_id", ["naive-100-0", "naive-500-100"])
def test_sample_documents_are_fully_covered(setup_id: str) -> None:
    setup = get_setup(setup_id)
    for document in get_template("jsonl-documents").load(SAMPLE):
        chunks = chunk_document(document, setup)
        covered: set[int] = set()
        for chunk in chunks:
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text
            covered.update(chunk.start_char + a for a, _ in token_spans(chunk.text))
        assert covered == {a for a, _ in token_spans(document.text)}
