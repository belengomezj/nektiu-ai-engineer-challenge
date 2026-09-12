"""FastAPI application for the NektiBot document assistant."""

import os
import time
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from api.citations import cited_answer
from api.messages import UNKNOWN_ANSWER, is_unknown_answer
from api.openai_gateway import create_answer, create_embeddings
from api.rag import HybridRetriever, SearchResult, split_markdown
from api.streaming import stream_response
from api.timing import elapsed_ms, log_duration, measure

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "sample.md"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

RETRIEVER_READY = False
RETRIEVER_ERROR: str | None = None
RETRIEVER_LAST_ATTEMPT = 0.0
HEALTH_RETRY_SECONDS = 10

app = FastAPI(title="Nektiu AI Engineer Challenge API", version="0.1.0")
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
    global RETRIEVER_ERROR, RETRIEVER_LAST_ATTEMPT, RETRIEVER_READY
    RETRIEVER_LAST_ATTEMPT = time.monotonic()
    try:
        retriever = HybridRetriever(split_markdown(load_document()), create_embeddings)
    except (OSError, OpenAIError, ValueError) as error:
        RETRIEVER_READY = False
        RETRIEVER_ERROR = type(error).__name__
        raise
    RETRIEVER_READY = True
    RETRIEVER_ERROR = None
    return retriever


def history_payload(request: ChatRequest) -> list[dict]:
    return [message.model_dump() for message in request.history]


def retrieval_question(request: ChatRequest) -> str:
    """Give short follow-ups the vocabulary from recent user questions."""
    previous = [message.content for message in request.history if message.role == "user"][-2:]
    return " ".join([*previous, request.question])


def search_documents(request: ChatRequest) -> list[SearchResult]:
    """Build the retriever if needed, then execute one query."""
    global RETRIEVER_ERROR, RETRIEVER_READY
    try:
        retriever = get_retriever()
    except (OSError, OpenAIError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Retriever is not ready") from error

    try:
        with measure("retrieval"):
            return retriever.search(retrieval_question(request))
    except OpenAIError as error:
        RETRIEVER_READY = False
        RETRIEVER_ERROR = type(error).__name__
        get_retriever.cache_clear()
        raise HTTPException(status_code=502, detail="Embedding provider unavailable") from error


@app.get("/api/health")
def health():
    retry_due = time.monotonic() - RETRIEVER_LAST_ATTEMPT >= HEALTH_RETRY_SECONDS
    if not RETRIEVER_READY and retry_due:
        with suppress(OSError, OpenAIError, ValueError):
            get_retriever()

    if not RETRIEVER_READY:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "retriever": False,
                "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
                "detail": RETRIEVER_ERROR or "not_initialized",
            },
        )
    return {
        "status": "ok",
        "retriever": RETRIEVER_READY,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "detail": None,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        results = search_documents(request)
        if not results:
            return ChatResponse(answer=UNKNOWN_ANSWER)

        sources = [result.chunk.content for result in results]
        with measure("generation"):
            raw_answer = create_answer(request.question, sources, history_payload(request))
        answer, cited_sources = cited_answer(raw_answer, sources)
        if is_unknown_answer(answer):
            return ChatResponse(answer=UNKNOWN_ANSWER)
        return ChatResponse(answer=answer, sources=cited_sources)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Unable to process the document") from error
    except OpenAIError as error:
        detail = "Language model provider unavailable"
        raise HTTPException(status_code=502, detail=detail) from error


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream model fragments and finish with the canonical answer and sources."""
    try:
        results = search_documents(request)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Unable to process the document") from error
    except OpenAIError as error:
        raise HTTPException(
            status_code=502, detail="Language model provider unavailable"
        ) from error

    sources = [result.chunk.content for result in results]
    return stream_response(request.question, history_payload(request), sources)
