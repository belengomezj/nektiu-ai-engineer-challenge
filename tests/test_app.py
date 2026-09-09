from fastapi.testclient import TestClient

import api.app as app_module
from api.rag import Chunk, SearchResult

client = TestClient(app_module.app)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, _question: str) -> list[SearchResult]:
        return self.results


def test_health() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_returns_grounded_answer_and_sources(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    results = [SearchResult(chunk, 1)]
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever(results))
    monkeypatch.setattr(
        app_module, "create_answer", lambda question, context: "Cuesta 49 € al mes."
    )

    response = client.post("/api/chat", json={"question": "¿Cuánto cuesta Starter?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Cuesta 49 € al mes.",
        "sources": ["Planes\nStarter cuesta 49 euros al mes."],
    }


def test_chat_says_unknown_without_calling_model(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever([]))
    model_called = False

    def fake_answer(_question: str, _context: list[str]) -> str:
        nonlocal model_called
        model_called = True
        return "Invented answer"

    monkeypatch.setattr(app_module, "create_answer", fake_answer)

    response = client.post("/api/chat", json={"question": "¿Quién ganó el mundial?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}
    assert model_called is False


def test_chat_hides_sources_when_model_cannot_answer(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    results = [SearchResult(chunk, 1)]
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever(results))
    monkeypatch.setattr(app_module, "create_answer", lambda question, context: "No lo sé.")

    response = client.post("/api/chat", json={"question": "¿Incluye llamadas?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}


def test_chat_normalizes_unknown_answer_without_period(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    monkeypatch.setattr(
        app_module, "get_retriever", lambda: FakeRetriever([SearchResult(chunk, 1)])
    )
    monkeypatch.setattr(app_module, "create_answer", lambda question, context: "  No lo sé.  ")

    response = client.post("/api/chat", json={"question": "¿Incluye llamadas?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}


def test_chat_rejects_blank_questions() -> None:
    response = client.post("/api/chat", json={"question": "   "})

    assert response.status_code == 422
