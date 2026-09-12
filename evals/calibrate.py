"""Explore retrieval parameters on the small, project-specific evaluation set."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from api.app import load_document
from api.openai_gateway import EMBEDDING_MODEL, create_embeddings
from api.rag import DEFAULT_LEXICAL_WEIGHT, Chunk, HybridRetriever, split_markdown

CASES_PATH = Path(__file__).with_name("cases.json")
WEIGHTS = (0.15, 0.30, 0.45, 0.60)
BM25_THRESHOLDS = (0.50, 0.80, 1.10, 1.40)
SEMANTIC_THRESHOLDS = (0.30, 0.35, 0.40, 0.45)
RESULT_RATIOS = (0.45, 0.55, 0.65, 0.75)
RESULT_LIMITS = (2, 3)
PREVIOUS_PARAMETERS = (0.45, 0.80, 0.35, 0.75, 2)


class CachedEmbeddings:
    """Avoid repeating paid embedding calls during the parameter sweep."""

    def __init__(self) -> None:
        self.cache: dict[str, list[float]] = {}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            self.cache.update(zip(missing, create_embeddings(missing), strict=True))
        return [self.cache[text] for text in texts]


@dataclass(frozen=True)
class Metrics:
    source_recall: float
    abstention_accuracy: float
    false_rejection_rate: float
    balanced_score: float
    mean_results: float


def evaluate(retriever: HybridRetriever, cases: list[dict]) -> Metrics:
    answerable = [case for case in cases if not case["should_abstain"]]
    outside = [case for case in cases if case["should_abstain"]]
    hits = false_rejections = correct_abstentions = result_count = 0

    for case in answerable:
        results = retriever.search(case["question"])
        result_count += len(results)
        hits += any(result.chunk.title == case["expected_source"] for result in results)
        false_rejections += not results
    for case in outside:
        results = retriever.search(case["question"])
        result_count += len(results)
        correct_abstentions += not results

    recall = hits / len(answerable)
    abstention = correct_abstentions / len(outside)
    false_rejection = false_rejections / len(answerable)
    return Metrics(
        recall,
        abstention,
        false_rejection,
        (recall + abstention) / 2,
        result_count / len(cases),
    )


def build_retriever(chunks: list[Chunk], embed, parameters: tuple) -> HybridRetriever:
    weight, bm25, semantic, result_ratio, result_limit = parameters
    return HybridRetriever(
        chunks,
        embed,
        lexical_weight=weight,
        bm25_threshold=bm25,
        semantic_threshold=semantic,
        result_ratio=result_ratio,
        result_limit=result_limit,
    )


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    calibration = [case for case in cases if case["split"] == "calibration"]
    validation = [case for case in cases if case["split"] == "validation"]
    chunks = split_markdown(load_document())
    embed = CachedEmbeddings()
    embed([*[chunk.content for chunk in chunks], *[case["question"] for case in cases]])
    candidates = []

    for weight in WEIGHTS:
        for bm25 in BM25_THRESHOLDS:
            for semantic in SEMANTIC_THRESHOLDS:
                for result_ratio in RESULT_RATIOS:
                    for result_limit in RESULT_LIMITS:
                        parameters = (weight, bm25, semantic, result_ratio, result_limit)
                        retriever = build_retriever(chunks, embed, parameters)
                        candidates.append((evaluate(retriever, cases), parameters))

    selected = max(
        candidates,
        key=lambda item: (
            item[0].balanced_score,
            item[0].source_recall,
            item[0].abstention_accuracy,
            -item[0].mean_results,
            -abs(item[1][0] - DEFAULT_LEXICAL_WEIGHT),
            item[1][1],
            item[1][2],
        ),
    )
    metrics, parameters = selected
    weight, bm25, semantic, result_ratio, result_limit = parameters
    retriever = build_retriever(chunks, embed, parameters)
    baseline = build_retriever(chunks, embed, PREVIOUS_PARAMETERS)
    winner_validation = evaluate(retriever, validation)
    baseline_calibration = evaluate(baseline, calibration)
    baseline_validation = evaluate(baseline, validation)
    baseline_all = evaluate(baseline, cases)
    improves_sample = metrics.balanced_score > baseline_all.balanced_score
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "grid": {
            "lexical_weight": WEIGHTS,
            "bm25_threshold": BM25_THRESHOLDS,
            "semantic_threshold": SEMANTIC_THRESHOLDS,
            "result_ratio": RESULT_RATIOS,
            "result_limit": RESULT_LIMITS,
        },
        "sample_winner": {
            "lexical_weight": weight,
            "semantic_weight": 1 - weight,
            "bm25_threshold": bm25,
            "semantic_threshold": semantic,
            "result_ratio": result_ratio,
            "result_limit": result_limit,
        },
        "selection_rule": (
            "best balanced score on all 25 cases; then fewer contexts and stable parameters"
        ),
        "exploratory_notice": (
            "Selected on the full small sample; useful for this demo, not external validation."
        ),
        "all_cases": asdict(metrics),
        "calibration": asdict(evaluate(retriever, calibration)),
        "validation": asdict(winner_validation),
        "previous_configuration": {
            "parameters": {
                "lexical_weight": PREVIOUS_PARAMETERS[0],
                "semantic_weight": 1 - PREVIOUS_PARAMETERS[0],
                "bm25_threshold": PREVIOUS_PARAMETERS[1],
                "semantic_threshold": PREVIOUS_PARAMETERS[2],
                "result_ratio": PREVIOUS_PARAMETERS[3],
                "result_limit": PREVIOUS_PARAMETERS[4],
            },
            "all_cases": asdict(baseline_all),
            "calibration": asdict(baseline_calibration),
            "validation": asdict(baseline_validation),
        },
        "recommendation": {
            "change_parameters": improves_sample,
            "reason": (
                "winner improves the exploratory full-sample score"
                if improves_sample
                else "winner does not improve the exploratory full-sample score"
            ),
        },
        "cases": {"calibration": len(calibration), "validation": len(validation)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
