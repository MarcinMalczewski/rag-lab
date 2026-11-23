import json
from pathlib import Path

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


def load(name: str) -> list[dict[str, object]]:
    lines = (SAMPLE / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_question_labels_point_at_the_stored_document_text() -> None:
    texts = {str(d["id"]): str(d["text"]) for d in load("documents.jsonl")}
    questions = load("questions.jsonl")

    assert len({q["id"] for q in questions}) == len(questions)
    for question in questions:
        relevant: list[dict[str, object]] = question["relevant"]  # type: ignore[assignment]
        assert bool(relevant) == (question["group"] != "G"), question["id"]
        for item in relevant:
            text = texts[str(item["document_id"])]
            start, end = int(str(item["start_char"])), int(str(item["end_char"]))
            assert 0 <= start < end <= len(text), question["id"]
            assert text[start:end] == item["text"], question["id"]
