"""Retrieval evaluation against labelled questions with answer spans."""

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_lab.chunker import chunk_document
from rag_lab.embeddings import Embedder
from rag_lab.provenance import ProvenanceStore
from rag_lab.retrieval import Passage, search_setup
from rag_lab.setups import Setup
from rag_lab.vector_store import QdrantIndex


@dataclass(frozen=True)
class Span:
    """Where an answer sits: a character range of one document."""

    document_id: str
    start_char: int
    end_char: int
    text: str | None = None


@dataclass(frozen=True)
class Question:
    """A labelled question; no spans means the corpus holds no answer."""

    id: str
    group: str
    text: str
    relevant: tuple[Span, ...]


@dataclass(frozen=True)
class QuestionResult:
    """Metrics and retrieved passages for one question."""

    question_id: str
    group: str
    answerable: bool
    recall: float | None
    precision: float | None
    first_relevant_rank: int | None
    top_score: float | None
    retrieved: list[dict[str, Any]]
    ndcg: float | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Everything an evaluation run produced."""

    run_id: str
    corpus_version: str
    top_k: int
    results: list[QuestionResult]
    summary: dict[str, Any]


def load_questions(path: Path) -> list[Question]:
    """Read a questions JSONL file, reporting the line of any problem."""
    questions: list[Question] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
            relevant = tuple(
                Span(
                    str(item["document_id"]),
                    int(item["start_char"]),
                    int(item["end_char"]),
                    item.get("text"),
                )
                for item in row["relevant"]
            )
            question = Question(
                str(row["id"]),
                str(row.get("group", "-")),
                str(row["question"]),
                relevant,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{where}: invalid question ({error!r})") from error
        if not question.text.strip():
            raise ValueError(f"{where}: empty question")
        if any(s.start_char < 0 or s.end_char <= s.start_char for s in relevant):
            raise ValueError(f"{where}: span needs 0 <= start_char < end_char")
        if question.id in seen:
            raise ValueError(f"{where}: duplicate id '{question.id}'")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


def check_labels(
    questions: list[Question], store: ProvenanceStore, corpus_version: str
) -> None:
    """Fail if a labelled span does not match the corpus that was indexed."""
    ids = sorted({s.document_id for q in questions for s in q.relevant})
    documents = store.get_documents(corpus_version, ids)
    for question in questions:
        for span in question.relevant:
            document = documents.get(span.document_id)
            if document is None:
                raise ValueError(
                    f"{question.id}: document '{span.document_id}' is not in the "
                    f"indexed corpus {corpus_version[:12]}."
                )
            found = document.text[span.start_char : span.end_char]
            if span.text is not None and found != span.text:
                raise ValueError(
                    f"{question.id}: span text does not match "
                    f"'{span.document_id}' in the indexed corpus."
                )


def overlaps(passage: Passage, span: Span) -> bool:
    """True when the passage and the span share at least one character."""
    return (
        passage.document_id == span.document_id
        and passage.start_char < span.end_char
        and span.start_char < passage.end_char
    )


def _dcg(flags: list[bool]) -> float:
    return sum(1 / math.log2(rank + 1) for rank, flag in enumerate(flags, 1) if flag)


def evaluate_question(
    question: Question,
    passages: list[Passage],
    relevant_chunks: int | None = None,
) -> QuestionResult:
    """Score one question's ranked passages against its answer spans.

    `relevant_chunks` is how many chunks of the corpus overlap an answer span; it
    sets the ideal ranking for nDCG. Without it, one chunk per span is assumed.
    """
    flags = [any(overlaps(p, s) for s in question.relevant) for p in passages]
    retrieved = [
        {
            "rank": p.rank,
            "document_id": p.document_id,
            "start_char": p.start_char,
            "end_char": p.end_char,
            "score": p.score,
            "relevant": flag,
        }
        for p, flag in zip(passages, flags, strict=True)
    ]
    top_score = passages[0].score if passages else None
    if not question.relevant:
        return QuestionResult(
            question.id, question.group, False, None, None, None, top_score, retrieved
        )
    covered = sum(any(overlaps(p, s) for p in passages) for s in question.relevant)
    ideal = _dcg([True] * min(len(passages), relevant_chunks or len(question.relevant)))
    first = next(
        (p.rank for p, flag in zip(passages, flags, strict=True) if flag), None
    )
    return QuestionResult(
        question.id,
        question.group,
        True,
        covered / len(question.relevant),
        sum(flags) / len(passages) if passages else 0.0,
        first,
        top_score,
        retrieved,
        _dcg(flags) / ideal if ideal else 0.0,
    )


def _metrics(results: list[QuestionResult]) -> dict[str, float | int]:
    count = len(results)
    return {
        "questions": count,
        "recall": sum(r.recall or 0.0 for r in results) / count,
        "precision": sum(r.precision or 0.0 for r in results) / count,
        "hit_rate": sum(r.first_relevant_rank is not None for r in results) / count,
        "mrr": sum(1 / r.first_relevant_rank for r in results if r.first_relevant_rank)
        / count,
        "ndcg": sum(r.ndcg or 0.0 for r in results) / count,
    }


def summarize(results: list[QuestionResult]) -> dict[str, Any]:
    """Aggregate answerable questions overall and per group, plus top scores."""
    answerable = [r for r in results if r.answerable]
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in answerable:
        groups[result.group].append(result)

    def mean_top(items: list[QuestionResult]) -> float | None:
        scores = [r.top_score for r in items if r.top_score is not None]
        return sum(scores) / len(scores) if scores else None

    return {
        "overall": _metrics(answerable) if answerable else None,
        "groups": {name: _metrics(items) for name, items in sorted(groups.items())},
        "mean_top_score_answerable": mean_top(answerable),
        "mean_top_score_no_answer": mean_top([r for r in results if not r.answerable]),
    }


def evaluate_setup(
    questions: list[Question],
    setup: Setup,
    embedder: Embedder,
    index: QdrantIndex,
    store: ProvenanceStore,
    top_k: int = 5,
) -> EvaluationReport:
    """Run every question against the setup's latest complete run."""
    run = store.latest_complete_run(setup.id)
    if run is None:
        raise ValueError(
            f"Setup '{setup.id}' has no complete run; run `raglab ingest` first."
        )
    run_id, corpus_version = run
    check_labels(questions, store, corpus_version)
    documents = store.get_documents(
        corpus_version, sorted({s.document_id for q in questions for s in q.relevant})
    )
    chunks = {i: chunk_document(d, setup) for i, d in documents.items()}
    results = []
    for question in questions:
        relevant_chunks = sum(
            any(
                chunk.start_char < span.end_char and span.start_char < chunk.end_char
                for span in question.relevant
                if span.document_id == document_id
            )
            for document_id, document_chunks in chunks.items()
            for chunk in document_chunks
        )
        passages = search_setup(
            question.text, setup, embedder, index, store, top_k
        ).passages
        results.append(evaluate_question(question, passages, relevant_chunks))
    return EvaluationReport(run_id, corpus_version, top_k, results, summarize(results))


def result_details(report: EvaluationReport) -> list[dict[str, Any]]:
    """Per-question results as JSON-friendly dictionaries."""
    return [asdict(result) for result in report.results]
