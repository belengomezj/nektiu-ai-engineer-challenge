"""Canonical public messages shared across API boundaries."""

UNKNOWN_ANSWER = "No lo sé"


def is_unknown_answer(answer: str) -> bool:
    """Accept the model's optional final period and normalize it at the API edge."""
    return answer.strip().rstrip(".").casefold() == UNKNOWN_ANSWER.casefold()
