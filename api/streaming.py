"""Server-sent events for incremental chat answers."""

import json

from fastapi.responses import StreamingResponse
from openai import OpenAIError

from api.citations import cited_answer
from api.messages import UNKNOWN_ANSWER, is_unknown_answer
from api.openai_gateway import create_answer_stream
from api.timing import measure

SOURCE_PREFIX = "FUENTES:"


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _visible_text(buffer: str) -> tuple[str, str, bool]:
    """Hold a possible citation prefix and hide it once complete."""
    upper = buffer.upper()
    if (position := upper.find(SOURCE_PREFIX)) >= 0:
        return buffer[:position].rstrip("\n"), "", True

    held = max(
        (size for size in range(1, len(SOURCE_PREFIX)) if SOURCE_PREFIX.startswith(upper[-size:])),
        default=0,
    )
    return buffer[:-held] if held else buffer, buffer[-held:] if held else "", False


def stream_response(question: str, history: list[dict], sources: list[str]) -> StreamingResponse:
    """Stream tokens followed by the normalized answer and its sources."""

    def events():
        if not sources:
            yield _event("done", {"answer": UNKNOWN_ANSWER, "sources": []})
            return

        fragments = []
        pending = ""
        marker_found = False
        try:
            with measure("generation.stream"):
                for text in create_answer_stream(question, sources, history):
                    fragments.append(text)
                    if marker_found:
                        continue
                    visible, pending, marker_found = _visible_text(pending + text)
                    if visible:
                        yield _event("token", {"text": visible})
        except OpenAIError:
            yield _event("error", {"message": "Language model provider unavailable"})
            return

        if pending and not marker_found:
            yield _event("token", {"text": pending})

        answer, response_sources = cited_answer("".join(fragments), sources)
        if is_unknown_answer(answer):
            answer, response_sources = UNKNOWN_ANSWER, []
        yield _event("done", {"answer": answer, "sources": response_sources})

    return StreamingResponse(events(), media_type="text/event-stream")
