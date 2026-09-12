"""Parse the source indices declared by the model."""

import logging
import re

LOG = logging.getLogger("nektibot")
SOURCE_LINE = re.compile(r"(?:^|\n)\s*FUENTES:\s*\[([^]]*)\]\s*$", re.IGNORECASE)


def cited_answer(raw_answer: str, sources: list[str]) -> tuple[str, list[str]]:
    """Return a clean answer and only its declared sources.

    Falling back to every retrieved source preserves the previous API contract when
    the model does not follow the lightweight citation protocol.
    """
    match = SOURCE_LINE.search(raw_answer)
    if not match:
        LOG.warning("citation_marker_missing")
        return raw_answer.strip(), sources

    try:
        indices = [int(value.strip()) for value in match.group(1).split(",") if value.strip()]
    except ValueError:
        LOG.warning("citation_marker_invalid")
        return raw_answer[: match.start()].strip(), sources

    if any(index < 1 or index > len(sources) for index in indices):
        LOG.warning("citation_index_invalid")
        return raw_answer[: match.start()].strip(), sources

    unique_indices = list(dict.fromkeys(indices))
    answer = raw_answer[: match.start()].strip()
    return answer, [sources[index - 1] for index in unique_indices]
