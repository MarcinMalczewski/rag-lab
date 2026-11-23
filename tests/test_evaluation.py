import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest

from conftest import FakeEmbedder, unique_id
from rag_lab.chunker import chunk_document
from rag_lab.documents import Document
from rag_lab.evaluation import (
    Question,
    Span,
    evaluate_question,
    evaluate_setup,
    load_questions,
    overlaps,
    result_details,
    summarize,
)
from rag_lab.indexing import index_chunks
from rag_lab.provenance import ProvenanceStore
from rag_lab.retrieval import Passage
from rag_lab.settings import get_settings
from rag_lab.setups import Setup, get_setup
from rag_lab.vector_store import QdrantIndex

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "questions.jsonl"


def passage(rank: int, doc: str, start: int, end: int, score: float = 0.5) -> Passage:
    return Passage(rank, score, f"c{rank}", doc, None, "", start, end, {})


def question(*spans: Span, group: str = "A") -> Question:
    return Question("q", group, "text", spans)


def test_overlap_needs_a_shared_character_in_the_same_document() -> None:
    span = Span("d", 10, 20)

    assert overlaps(passage(1, "d", 0, 11), span)
    assert overlaps(passage(1, "d", 19, 30), span)
    assert not overlaps(passage(1, "d", 0, 10), span)  # touching only
    assert not overlaps(passage(1, "d", 20, 30), span)
    assert not overlaps(passage(1, "other", 0, 30), span)


def test_recall_precision_and_first_rank() -> None:
    result = evaluate_question(
        question(Span("d", 0, 10), Span("e", 0, 10)),
        [passage(1, "x", 0, 5), passage(2, "d", 5, 8), passage(3, "d", 9, 20)],
    )

    assert result.recall == 0.5  # span in "e" is never retrieved
    assert result.precision == pytest.approx(2 / 3)
    assert result.first_relevant_rank == 2


def test_ndcg_is_measured_against_the_ideal_ranking() -> None:
    spans = (Span("d", 0, 10), Span("e", 0, 10))
    top = passage(1, "d", 0, 5), passage(2, "e", 0, 5)
    miss = passage(3, "x", 0, 5)

    perfect = evaluate_question(question(*spans), [*top, miss], relevant_chunks=2)
    ranks_1_and_3 = evaluate_question(
        question(*spans), [top[0], miss, passage(3, "e", 0, 5)], relevant_chunks=2
    )
    nothing = evaluate_question(question(*spans), [miss], relevant_chunks=2)

    assert perfect.ndcg == pytest.approx(1.0)
    assert ranks_1_and_3.ndcg == pytest.approx((1 + 0.5) / (1 + 1 / 1.5849625))
    assert nothing.ndcg == 0.0


def test_one_relevant_chunk_found_first_gives_ndcg_one_despite_k() -> None:
    result = evaluate_question(
        question(Span("d", 0, 10)),
        [passage(i, "d" if i == 1 else "x", 0, 5) for i in range(1, 6)],
        relevant_chunks=1,
    )

    assert result.precision == 0.2 and result.ndcg == pytest.approx(1.0)


def test_nothing_relevant_retrieved() -> None:
    result = evaluate_question(question(Span("d", 0, 10)), [passage(1, "x", 0, 5)])

    assert (result.recall, result.precision) == (0.0, 0.0)
    assert result.first_relevant_rank is None


def test_no_answer_question_only_records_top_score() -> None:
    result = evaluate_question(question(), [passage(1, "d", 0, 5, score=0.7)])

    assert not result.answerable
    assert (result.recall, result.first_relevant_rank, result.top_score) == (
        None,
        None,
        0.7,
    )


def test_summary_by_group_excludes_no_answer_questions() -> None:
    hit = evaluate_question(question(Span("d", 0, 5)), [passage(1, "d", 0, 5, 0.9)])
    late = evaluate_question(
        question(Span("d", 0, 5), group="B"),
        [passage(1, "x", 0, 5), passage(2, "d", 0, 5)],
    )
    none = evaluate_question(question(group="G"), [passage(1, "d", 0, 5, 0.3)])

    summary = summarize([hit, late, none])

    assert summary["overall"]["questions"] == 2
    assert summary["overall"]["mrr"] == pytest.approx(0.75)  # (1 + 1/2) / 2
    assert summary["groups"]["A"]["hit_rate"] == 1.0
    assert summary["groups"]["B"]["mrr"] == 0.5
    assert summary["overall"]["ndcg"] == pytest.approx((1 + 0.6309298) / 2)
    assert "G" not in summary["groups"]
    assert summary["mean_top_score_no_answer"] == 0.3


def test_load_sample_questions() -> None:
    questions = load_questions(SAMPLE)

    assert len(questions) == 25
    assert [q.id for q in questions if not q.relevant] == ["sample-G1", "sample-G2"]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", r":1: invalid question"),
        ('{"id": "a", "question": "q"}', r"invalid question"),
        ('{"id": "a", "question": " ", "relevant": []}', r"empty question"),
        (
            '{"id": "a", "question": "q", "relevant":'
            ' [{"document_id": "d", "start_char": 5, "end_char": 5}]}',
            r"0 <= start_char < end_char",
        ),
    ],
)
def test_invalid_questions_report_the_line(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_questions(path)


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    row = '{"id": "a", "question": "q", "relevant": []}\n'
    path = tmp_path / "q.jsonl"
    path.write_text(row * 2, encoding="utf-8")

    with pytest.raises(ValueError, match=":2: duplicate id 'a'"):
        load_questions(path)


EMBEDDER = FakeEmbedder(768)
DOCUMENT = Document("a", " ".join(f"alpha{i}" for i in range(250)), "Doc A", {})


@pytest.fixture
def store(postgres_settings: None) -> Iterator[ProvenanceStore]:
    settings = get_settings()
    with ProvenanceStore(settings.postgres_dsn, settings.provenance_schema) as store:
        yield store


@pytest.fixture
def indexed(
    store: ProvenanceStore, qdrant_url: str
) -> tuple[Setup, QdrantIndex, ProvenanceStore, str]:
    setup = dataclasses.replace(get_setup("nomic-naive-100-0"), id=unique_id("setup"))
    index = QdrantIndex(qdrant_url)
    corpus = unique_id("corpus")
    store.register_setup(setup)
    run_id = store.start_run(setup.id, "test", corpus)
    store.save_documents(corpus, [DOCUMENT])
    chunks = chunk_document(DOCUMENT, setup)
    result = index_chunks(chunks, setup, run_id, corpus, EMBEDDER, index)
    store.add_vector_refs(run_id, result.collection, result.point_ids)
    store.finish_run(run_id)
    return setup, index, store, run_id


def test_evaluate_setup_finds_the_labelled_chunk(
    indexed: tuple[Setup, QdrantIndex, ProvenanceStore, str],
) -> None:
    setup, index, store, run_id = indexed
    chunk = chunk_document(DOCUMENT, setup)[1]
    inside = Span("a", chunk.start_char + 5, chunk.start_char + 40)
    questions = [
        Question("hit", "A", chunk.text, (inside,)),  # query is the chunk itself
        Question("none", "G", "unrelated", ()),
    ]

    report = evaluate_setup(questions, setup, EMBEDDER, index, store, top_k=3)

    hit, none = report.results
    assert report.run_id == run_id
    assert (hit.recall, hit.first_relevant_rank) == (1.0, 1)
    assert hit.ndcg == pytest.approx(1.0)
    assert none.answerable is False and none.top_score is not None
    assert report.summary["overall"]["hit_rate"] == 1.0
    assert len(result_details(report)) == 2


def test_labels_that_do_not_match_the_corpus_are_rejected(
    indexed: tuple[Setup, QdrantIndex, ProvenanceStore, str],
) -> None:
    setup, index, store, _ = indexed
    wrong_text = Question("q", "A", "t", (Span("a", 0, 5, "not the text"),))
    wrong_doc = Question("q", "A", "t", (Span("missing", 0, 5),))

    with pytest.raises(ValueError, match="span text does not match"):
        evaluate_setup([wrong_text], setup, EMBEDDER, index, store)
    with pytest.raises(ValueError, match="'missing' is not in the indexed corpus"):
        evaluate_setup([wrong_doc], setup, EMBEDDER, index, store)


def test_setup_without_complete_run_is_rejected(
    store: ProvenanceStore, qdrant_url: str
) -> None:
    setup = dataclasses.replace(get_setup("nomic-naive-100-0"), id=unique_id("s"))
    store.register_setup(setup)

    with pytest.raises(ValueError, match="no complete run"):
        evaluate_setup([question()], setup, EMBEDDER, QdrantIndex(qdrant_url), store)


def test_evaluation_is_saved_with_the_run(
    indexed: tuple[Setup, QdrantIndex, ProvenanceStore, str],
) -> None:
    setup, index, store, run_id = indexed
    report = evaluate_setup(
        [Question("q", "A", "alpha1", (Span("a", 0, 5),))],
        setup,
        EMBEDDER,
        index,
        store,
    )

    evaluation_id = store.save_evaluation(
        run_id, "q.jsonl", "sum", 5, report.summary, result_details(report)
    )

    (saved,) = store.load_evaluations(run_id)
    assert saved["evaluation_id"] == evaluation_id
    assert saved["summary"]["overall"]["questions"] == 1
    assert saved["details"][0]["question_id"] == "q"
