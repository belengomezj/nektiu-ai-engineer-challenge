"""FastAPI application for the NektiBot document assistant."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from api.messages import UNKNOWN_ANSWER, is_unknown_answer
from api.openai_gateway import create_answer, create_embeddings
from api.rag import HybridRetriever, split_markdown

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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        results = get_retriever().search(request.question)
        if not results:
            return ChatResponse(answer=UNKNOWN_ANSWER)

        sources = [result.chunk.content for result in results]
        answer = create_answer(request.question, sources).strip()
        if is_unknown_answer(answer):
            return ChatResponse(answer=UNKNOWN_ANSWER)
        return ChatResponse(answer=answer, sources=sources)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Unable to process the document") from error
    except OpenAIError as error:
        detail = "Language model provider unavailable"
        raise HTTPException(status_code=502, detail=detail) from error
