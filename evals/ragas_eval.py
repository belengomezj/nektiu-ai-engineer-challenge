"""Evaluate the answerable cases with Ragas against a running API."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from api.app import get_retriever
from api.messages import UNKNOWN_ANSWER
from api.openai_gateway import create_answer

CASES_PATH = Path(__file__).with_name("cases.json")
load_dotenv()
load_dotenv(CASES_PATH.parent.parent / "api" / ".env")
MODEL = os.getenv("RAGAS_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")


def collect_samples() -> list[dict]:
    """Run the RAG pipeline and retain retrieval context even if it abstains."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    samples = []
    retriever = get_retriever()
    for case in cases:
        if case["should_abstain"]:
            continue
        results = retriever.search(case["question"])
        contexts = [result.chunk.content for result in results]
        answer = create_answer(case["question"], contexts) if contexts else UNKNOWN_ANSWER
        samples.append(
            {
                "user_input": case["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": case["reference"],
            }
        )
    return samples


async def evaluate(samples: list[dict]) -> None:
    """Print per-case scores and their averages; Ragas supplies the evaluators."""
    client = AsyncOpenAI()
    llm = llm_factory(MODEL, client=client)
    embeddings = embedding_factory("openai", model=EMBEDDING_MODEL, client=client)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    scores = {name: [] for name in metrics}

    try:
        for sample in samples:
            values = {
                "faithfulness": await metrics["faithfulness"].ascore(
                    user_input=sample["user_input"],
                    response=sample["response"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ),
                "answer_relevancy": await metrics["answer_relevancy"].ascore(
                    user_input=sample["user_input"], response=sample["response"]
                ),
                "context_precision": await metrics["context_precision"].ascore(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ),
                "context_recall": await metrics["context_recall"].ascore(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                ),
            }
            numeric = {name: float(result.value) for name, result in values.items()}
            for name, value in numeric.items():
                scores[name].append(value)
            print(f"\n{sample['user_input']}")
            print("  " + "  ".join(f"{name}={value:.2f}" for name, value in numeric.items()))
    finally:
        await client.close()

    print("\nAverage")
    for name, values in scores.items():
        print(f"  {name}: {sum(values) / len(values):.2f}")


if __name__ == "__main__":
    asyncio.run(evaluate(collect_samples()))
