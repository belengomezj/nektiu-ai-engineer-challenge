import numpy as np

from api.rag import BM25Index, Chunk, HybridRetriever, split_markdown, tokenize


def test_split_markdown_creates_one_chunk_per_section() -> None:
    document = "# Manual\n\n## First\nOne.\n\n## Second\nTwo."

    chunks = split_markdown(document)

    assert chunks == [Chunk("First", "One."), Chunk("Second", "Two.")]


def test_tokenize_normalizes_accents_and_removes_stop_words() -> None:
    assert tokenize("¿Cuál es el precio de la suscripción?") == ["precio", "suscripcion"]


def test_bm25_ranks_exact_terms_first() -> None:
    index = BM25Index(["Planes y precios Starter", "Privacidad y cifrado de datos"])

    scores = index.scores("precio del plan Starter")

    assert scores[0] > scores[1]


def test_hybrid_retriever_uses_semantic_similarity() -> None:
    chunks = [Chunk("Planes", "Starter cuesta 49 euros."), Chunk("Datos", "Datos cifrados.")]
    embeddings = {
        chunks[0].content: [1.0, 0.0, 0.0],
        chunks[1].content: [0.0, 1.0, 0.0],
        "¿Cuánto tengo que pagar?": [1.0, 0.0, 0.0],
    }
    retriever = HybridRetriever(chunks, lambda texts: [embeddings[text] for text in texts])

    results = retriever.search("¿Cuánto tengo que pagar?")

    assert results[0].chunk.title == "Planes"


def test_hybrid_retriever_rejects_unrelated_question() -> None:
    chunks = [Chunk("Planes", "Starter cuesta 49 euros."), Chunk("Datos", "Datos cifrados.")]
    vectors = np.eye(3).tolist()
    embeddings = {
        chunks[0].content: vectors[0],
        chunks[1].content: vectors[1],
        "¿Quién ganó el mundial?": vectors[2],
    }
    retriever = HybridRetriever(chunks, lambda texts: [embeddings[text] for text in texts])

    assert retriever.search("¿Quién ganó el mundial?") == []
