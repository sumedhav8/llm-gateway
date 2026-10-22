import httpx
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


# --- Day 4 tests ---------------------------------------------------------

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_without_question_is_rejected():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_with_wrong_type_is_rejected():
    response = client.post("/ask", json={"question": 123})
    assert response.status_code == 422


def test_ask_when_model_unreachable_returns_503(monkeypatch):
    # Point the gateway at an address where nothing is running
    monkeypatch.setattr(main, "MODEL_BASE_URL", "http://127.0.0.1:9/v1")
    response = client.post("/ask", json={"question": "Hello?"})
    assert response.status_code == 503
    assert "Is Ollama running" in response.json()["detail"]


# --- Day 5 tests: a fake model, so tests don't need Ollama ---------------

def use_fake_model(monkeypatch, status_code, body):
    """Replace the real AI model with a fake that always gives the same reply."""

    def fake_model(request):
        return httpx.Response(status_code, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(fake_model), **kwargs),
    )


def test_ask_returns_answer_and_token_counts(monkeypatch):
    use_fake_model(
        monkeypatch,
        200,
        {
            "choices": [{"message": {"content": "Paris"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )

    response = client.post("/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris"
    assert data["model"] == main.MODEL_NAME
    assert data["input_tokens"] == 12
    assert data["output_tokens"] == 3
    assert data["latency_ms"] >= 0


def test_ask_when_model_returns_error_gives_502(monkeypatch):
    use_fake_model(monkeypatch, 404, {"error": {"message": "model not found"}})

    response = client.post("/ask", json={"question": "Hello?"})

    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]