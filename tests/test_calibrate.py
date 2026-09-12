import json
from pathlib import Path

import pytest

from api.rag import Chunk, SearchResult, split_markdown
from evals.calibrate import CASES_PATH, evaluate


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results

    def search(self, question: str) -> list[SearchResult]:
        return [SearchResult(Chunk(title, "text"), 1.0) for title in self.results[question]]


def test_evaluate_keeps_retrieval_metrics_separate() -> None:
    cases = [
        {"question": "a", "expected_source": "A", "should_abstain": False},
        {"question": "b", "expected_source": "B", "should_abstain": False},
        {"question": "c", "expected_source": "C", "should_abstain": False},
        {"question": "d", "expected_source": None, "should_abstain": True},
        {"question": "e", "expected_source": None, "should_abstain": True},
    ]
    retriever = FakeRetriever({"a": ["A"], "b": ["B"], "c": [], "d": [], "e": ["A"]})

    metrics = evaluate(retriever, cases)

    assert metrics.source_recall == pytest.approx(2 / 3)
    assert metrics.abstention_accuracy == 0.5
    assert metrics.false_rejection_rate == pytest.approx(1 / 3)
    assert metrics.balanced_score == pytest.approx(7 / 12)


def test_dataset_has_separate_splits_and_valid_source_labels() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    document = Path("data/sample.md").read_text(encoding="utf-8")
    titles = {chunk.title for chunk in split_markdown(document)}

    assert len(cases) == 25
    assert {case["split"] for case in cases} == {"calibration", "validation"}
    assert {case["expected_source"] for case in cases if case["expected_source"]} <= titles
    assert sum(case["should_abstain"] for case in cases) == 5
