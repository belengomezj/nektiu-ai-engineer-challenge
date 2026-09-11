"""FastAPI application for the NektiBot document assistant."""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from api.messages import UNKNOWN_ANSWER, is_unknown_answer
from api.openai_gateway import create_answer, create_embeddings
from api.rag import HybridRetriever, split_markdown
from api.streaming import stream_response
from api.timing import elapsed_ms, log_duration, measure

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "sample.md"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app = FastAPI(title="Nektiu AI Engineer Challenge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_server_timing(request, call_next):
    """Expose total server time and log it without recording user content."""
    started = time.perf_counter()
    response = await call_next(request)
    duration = elapsed_ms(started)
    response.headers["Server-Timing"] = f"app;dur={duration}"
    log_duration("http", duration)
    return response


class HistoryMessage(BaseModel):
    role: Literal["assistant", "user"]
    content: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


def load_document() -> str:
    return DATA_PATH.read_text(encoding="utf-8")


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(split_markdown(load_document()), create_embeddings)


def history_payload(request: ChatRequest) -> list[dict]:
    return [message.model_dump() for message in request.history]


def retrieval_question(request: ChatRequest) -> str:
    """Give short follow-ups the vocabulary from recent user questions."""
    previous = [message.content for message in request.history if message.role == "user"][-2:]
    return " ".join([*previous, request.question])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        with measure("retrieval"):
            results = get_retriever().search(retrieval_question(request))
        if not results:
            return ChatResponse(answer=UNKNOWN_ANSWER)

        sources = [result.chunk.content for result in results]
        with measure("generation"):
            answer = create_answer(request.question, sources, history_payload(request)).strip()
        if is_unknown_answer(answer):
            return ChatResponse(answer=UNKNOWN_ANSWER)
        return ChatResponse(answer=answer, sources=sources)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Unable to process the document") from error
    except OpenAIError as error:
        detail = "Language model provider unavailable"
        raise HTTPException(status_code=502, detail=detail) from error


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream model fragments and finish with the canonical answer and sources."""
    try:
        with measure("retrieval"):
            results = get_retriever().search(retrieval_question(request))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Unable to process the document") from error
    except OpenAIError as error:
        raise HTTPException(
            status_code=502, detail="Language model provider unavailable"
        ) from error

    sources = [result.chunk.content for result in results]
    return stream_response(request.question, history_payload(request), sources)
