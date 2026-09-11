"""Server-sent events for incremental chat answers."""

import json

from fastapi.responses import StreamingResponse
from openai import OpenAIError

from api.messages import UNKNOWN_ANSWER, is_unknown_answer
from api.openai_gateway import create_answer_stream
from api.timing import measure


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_response(question: str, history: list[dict], sources: list[str]) -> StreamingResponse:
    """Stream tokens followed by the normalized answer and its sources."""

    def events():
        if not sources:
            yield _event("done", {"answer": UNKNOWN_ANSWER, "sources": []})
            return

        fragments = []
        try:
            with measure("generation.stream"):
                for text in create_answer_stream(question, sources, history):
                    fragments.append(text)
                    yield _event("token", {"text": text})
        except OpenAIError:
            yield _event("error", {"message": "Language model provider unavailable"})
            return

        answer = "".join(fragments).strip()
        response_sources = sources
        if is_unknown_answer(answer):
            answer, response_sources = UNKNOWN_ANSWER, []
        yield _event("done", {"answer": answer, "sources": response_sources})

    return StreamingResponse(events(), media_type="text/event-stream")
