import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

import api.app as app_module
import api.streaming as streaming_module
from api.rag import Chunk, SearchResult

client = TestClient(app_module.app)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, _question: str) -> list[SearchResult]:
        return self.results


def test_health(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_retriever", lambda: object())
    monkeypatch.setattr(app_module, "RETRIEVER_READY", True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "retriever": True,
        "openai_configured": True,
        "detail": None,
    }
    assert response.headers["server-timing"].startswith("app;dur=")


def test_health_reports_degraded_retriever(monkeypatch) -> None:
    def fail():
        raise OpenAIError("temporary provider failure")

    monkeypatch.setattr(app_module, "get_retriever", fail)
    monkeypatch.setattr(app_module, "RETRIEVER_ERROR", "OpenAIError")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "retriever": False,
        "openai_configured": False,
        "detail": "OpenAIError",
    }


def test_health_throttles_provider_retries(monkeypatch) -> None:
    calls = 0

    def count_call():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(app_module, "get_retriever", count_call)
    monkeypatch.setattr(app_module, "RETRIEVER_READY", False)
    monkeypatch.setattr(app_module, "RETRIEVER_ERROR", "OpenAIError")
    monkeypatch.setattr(app_module, "RETRIEVER_LAST_ATTEMPT", app_module.time.monotonic())

    response = client.get("/api/health")

    assert response.status_code == 503
    assert calls == 0


def test_retriever_retries_after_a_failed_build(monkeypatch) -> None:
    calls = 0

    def flaky_embeddings(texts):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OpenAIError("temporary")
        return [[1.0, 0.0]] * len(texts)

    app_module.get_retriever.cache_clear()
    monkeypatch.setattr(app_module, "load_document", lambda: "## Planes\nStarter cuesta 49 euros.")
    monkeypatch.setattr(app_module, "create_embeddings", flaky_embeddings)
    try:
        with pytest.raises(OpenAIError):
            app_module.get_retriever()
        assert app_module.get_retriever() is app_module.get_retriever()
        assert calls == 2
    finally:
        app_module.get_retriever.cache_clear()


def test_chat_returns_grounded_answer_and_sources(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    results = [SearchResult(chunk, 1)]
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever(results))
    monkeypatch.setattr(
        app_module,
        "create_answer",
        lambda question, context, history=None: "Cuesta 49 € al mes.\nFUENTES: [1]",
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

    def fake_answer(_question: str, _context: list[str], _history=None) -> str:
        nonlocal model_called
        model_called = True
        return "Invented answer"

    monkeypatch.setattr(app_module, "create_answer", fake_answer)

    response = client.post("/api/chat", json={"question": "¿Quién ganó el mundial?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}
    assert model_called is False


def test_chat_returns_503_while_retriever_is_unavailable(monkeypatch) -> None:
    def fail():
        raise OpenAIError("temporary")

    monkeypatch.setattr(app_module, "get_retriever", fail)

    response = client.post("/api/chat", json={"question": "¿Qué precio se indica?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Retriever is not ready"}


def test_embedding_failure_marks_health_as_degraded(monkeypatch) -> None:
    class FailingRetriever:
        def search(self, question):
            raise OpenAIError("temporary")

    def get_failing_retriever():
        return FailingRetriever()

    get_failing_retriever.cache_clear = lambda: None
    monkeypatch.setattr(app_module, "get_retriever", get_failing_retriever)
    monkeypatch.setattr(app_module, "RETRIEVER_READY", True)

    response = client.post("/api/chat", json={"question": "¿Qué precio se indica?"})

    assert response.status_code == 502
    assert app_module.RETRIEVER_READY is False
    assert app_module.RETRIEVER_ERROR == "OpenAIError"


def test_chat_hides_sources_when_model_cannot_answer(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    results = [SearchResult(chunk, 1)]
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever(results))
    monkeypatch.setattr(
        app_module, "create_answer", lambda question, context, history=None: "No lo sé."
    )

    response = client.post("/api/chat", json={"question": "¿Incluye llamadas?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}


def test_chat_normalizes_unknown_answer_without_period(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    monkeypatch.setattr(
        app_module, "get_retriever", lambda: FakeRetriever([SearchResult(chunk, 1)])
    )
    monkeypatch.setattr(
        app_module,
        "create_answer",
        lambda question, context, history=None: "  No lo sé.  ",
    )

    response = client.post("/api/chat", json={"question": "¿Incluye llamadas?"})

    assert response.json() == {"answer": "No lo sé", "sources": []}


def test_chat_rejects_blank_questions() -> None:
    response = client.post("/api/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_forwards_a_short_history(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    monkeypatch.setattr(
        app_module, "get_retriever", lambda: FakeRetriever([SearchResult(chunk, 1)])
    )
    captured = []

    def fake_answer(_question, _sources, history):
        captured.extend(history)
        return "El Business cuesta 149 € al mes."

    monkeypatch.setattr(app_module, "create_answer", fake_answer)
    response = client.post(
        "/api/chat",
        json={
            "question": "¿Y el Business?",
            "history": [
                {"role": "user", "content": "¿Cuánto cuesta Starter?"},
                {"role": "assistant", "content": "Cuesta 49 € al mes."},
            ],
        },
    )

    assert response.status_code == 200
    assert [message["role"] for message in captured] == ["user", "assistant"]


def test_stream_returns_tokens_and_final_sources(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    monkeypatch.setattr(
        app_module, "get_retriever", lambda: FakeRetriever([SearchResult(chunk, 1)])
    )
    monkeypatch.setattr(
        streaming_module,
        "create_answer_stream",
        lambda *args: iter(["Cuesta ", "49 €.\n", "FUENTES: [1]"]),
    )

    response = client.post("/api/chat/stream", json={"question": "¿Qué precio se indica?"})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert '"answer": "Cuesta 49 €."' in response.text
    assert "Planes\\nStarter" in response.text
    assert "FUENTES:" not in response.text


def test_stream_normalizes_unknown_and_hides_sources(monkeypatch) -> None:
    chunk = Chunk("Planes", "Starter cuesta 49 euros al mes.")
    monkeypatch.setattr(
        app_module, "get_retriever", lambda: FakeRetriever([SearchResult(chunk, 1)])
    )
    monkeypatch.setattr(streaming_module, "create_answer_stream", lambda *args: iter(["No lo sé."]))

    response = client.post("/api/chat/stream", json={"question": "¿Incluye llamadas?"})

    assert f'"answer": "{app_module.UNKNOWN_ANSWER}"' in response.text
    assert '"sources": []' in response.text
