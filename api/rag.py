"""Document chunking and hybrid retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EmbeddingFunction = Callable[[list[str]], list[list[float]]]

TOKEN_PATTERN = re.compile(r"[a-z0-9@.€]+")
STOP_WORDS = {
    "a",
    "al",
    "con",
    "cual",
    "de",
    "del",
    "el",
    "en",
    "es",
    "esta",
    "hay",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "su",
    "un",
    "una",
    "y",
}


@dataclass(frozen=True)
class Chunk:
    """A retrievable Markdown section."""

    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class SearchResult:
    """A chunk and its hybrid relevance score."""

    chunk: Chunk
    score: float


def split_markdown(document: str) -> list[Chunk]:
    """Split a Markdown document at second-level headings."""
    chunks: list[Chunk] = []
    title: str | None = None
    body: list[str] = []

    for line in document.splitlines():
        if line.startswith("## "):
            if title and body:
                chunks.append(Chunk(title, "\n".join(body).strip()))
            title = line.removeprefix("## ").strip()
            body = []
        elif title and line.strip():
            body.append(line.strip())

    if title and body:
        chunks.append(Chunk(title, "\n".join(body).strip()))
    return chunks


def tokenize(text: str) -> list[str]:
    """Normalize Spanish text into useful retrieval terms."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return [token for token in TOKEN_PATTERN.findall(ascii_text) if token not in STOP_WORDS]


class BM25Index:
    """Small in-memory BM25 index."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.average_length = sum(map(len, self.documents)) / max(len(self.documents), 1)
        self.document_frequency = self._document_frequency()

    def _document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for document in self.documents:
            frequency.update(set(document))
        return frequency

    def scores(self, query: str) -> np.ndarray:
        query_terms = set(tokenize(query))
        return np.array([self._score(document, query_terms) for document in self.documents])

    def _score(self, document: list[str], query_terms: set[str]) -> float:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            document_count = len(self.documents)
            term_count = self.document_frequency[term]
            numerator_idf = document_count - term_count + 0.5
            inverse_frequency = math.log(1 + numerator_idf / (term_count + 0.5))
            length_ratio = len(document) / max(self.average_length, 1)
            numerator = frequencies[term] * (self.k1 + 1)
            denominator = frequencies[term] + self.k1 * (1 - self.b + self.b * length_ratio)
            score += inverse_frequency * numerator / denominator
        return score


class HybridRetriever:
    """Combine lexical BM25 and semantic cosine similarity."""

    def __init__(self, chunks: list[Chunk], embed: EmbeddingFunction) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required")
        self.chunks = chunks
        self.embed = embed
        self.bm25 = BM25Index([chunk.content for chunk in chunks])
        self.embeddings = _normalize_rows(np.asarray(embed([chunk.content for chunk in chunks])))

    def search(self, question: str, limit: int = 2) -> list[SearchResult]:
        lexical = self.bm25.scores(question)
        query_embedding = _normalize_rows(np.asarray(self.embed([question])))[0]
        semantic = self.embeddings @ query_embedding

        if lexical.max(initial=0) < 0.8 and semantic.max(initial=0) < 0.35:
            return []

        hybrid = 0.45 * _relative(lexical) + 0.55 * _relative(semantic)
        minimum_score = max(0.25, float(hybrid.max()) * 0.75)
        indices = [index for index in np.argsort(hybrid)[::-1] if hybrid[index] >= minimum_score]
        indices = indices[:limit]
        return [SearchResult(self.chunks[index], float(hybrid[index])) for index in indices]


def _relative(scores: np.ndarray) -> np.ndarray:
    maximum = scores.max(initial=0)
    return scores / maximum if maximum > 0 else np.zeros_like(scores)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero")
    return values / norms
