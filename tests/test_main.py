from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


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