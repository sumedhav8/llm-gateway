import json
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, main, models, pricing

client = TestClient(main.app)

PARIS_REPLY = {
    "choices": [{"message": {"content": "Paris"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


# --- Fixtures: run every test with fake settings and no real database ----

@pytest.fixture(autouse=True)
def test_models(monkeypatch):
    """Use known test models, whatever is in your .env file."""
    monkeypatch.setitem(config.MODELS, "cheap",
                        config.ModelConfig("cheap", "http://cheap.test/v1", "ollama", "cheap-model"))
    monkeypatch.setitem(config.MODELS, "strong",
                        config.ModelConfig("strong", "http://strong.test/v1", config.PLACEHOLDER_KEY, "strong-model"))


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Catch rows the gateway would save, instead of writing to the real database."""
    rows = []
    monkeypatch.setattr(main.db, "log_request", lambda **row: rows.append(row))
    return rows


# --- Helpers ---------------------------------------------------------------

def use_fake_model(monkeypatch, status_code=200, body=PARIS_REPLY):
    """Replace real AI models with a fake. Returns a list of the requests it received."""
    sent = []

    def fake_model(request):
        sent.append({"url": str(request.url), "json": json.loads(request.content)})
        return httpx.Response(status_code, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(fake_model), **kwargs),
    )
    return sent


def use_slow_model(monkeypatch):
    """Replace real AI models with one that always times out."""

    def slow_model(request):
        raise httpx.ReadTimeout("The model took too long", request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(slow_model), **kwargs),
    )


# --- Basic endpoint tests ------------------------------------------------

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


def test_ask_with_unknown_tier_is_rejected():
    response = client.post("/ask", json={"question": "Hi", "tier": "medium"})
    assert response.status_code == 422


# --- Cheap model (the default) ----------------------------------------------

def test_ask_returns_answer_and_token_counts(monkeypatch):
    use_fake_model(monkeypatch)

    response = client.post("/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris"
    assert data["model"] == "cheap-model"
    assert data["tier"] == "cheap"
    assert data["input_tokens"] == 12
    assert data["output_tokens"] == 3
    assert data["latency_ms"] >= 0


def test_ask_when_model_unreachable_returns_503(monkeypatch, logged):
    monkeypatch.setitem(config.MODELS, "cheap",
                        replace(config.MODELS["cheap"], base_url="http://127.0.0.1:9/v1"))

    response = client.post("/ask", json={"question": "Hello?"})

    assert response.status_code == 503
    assert "Can't reach the cheap model" in response.json()["detail"]
    assert logged[0]["status_code"] == 503
    assert logged[0]["answer"] is None


def test_ask_when_model_returns_error_gives_502(monkeypatch):
    use_fake_model(monkeypatch, 404, {"error": {"message": "model not found"}})

    response = client.post("/ask", json={"question": "Hello?"})

    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]


def test_ask_when_model_times_out_returns_504(monkeypatch, logged):
    use_slow_model(monkeypatch)

    response = client.post("/ask", json={"question": "Explain how vaccines work"})

    assert response.status_code == 504
    assert logged[0]["status_code"] == 504


def test_successful_ask_is_logged_with_cost(monkeypatch, logged):
    use_fake_model(monkeypatch)
    monkeypatch.setitem(pricing.PRICES_PER_MILLION, "cheap-model", (Decimal("1"), Decimal("2")))

    client.post("/ask", json={"question": "What is the capital of France?"})

    assert len(logged) == 1
    row = logged[0]
    assert row["status_code"] == 200
    assert row["question"] == "What is the capital of France?"
    assert row["answer"] == "Paris"
    assert row["model"] == "cheap-model"
    assert row["route"] == "cheap"
    # 12 input tokens x $1/M + 3 output tokens x $2/M = $0.000018
    assert row["cost_usd"] == Decimal("0.000018")



# --- Strong model (Day 12) ------------------------------------------------

def give_strong_model_a_key(monkeypatch):
    monkeypatch.setitem(config.MODELS, "strong",
                        replace(config.MODELS["strong"], api_key="test-key"))


def test_strong_tier_uses_the_strong_model(monkeypatch):
    give_strong_model_a_key(monkeypatch)
    sent = use_fake_model(monkeypatch)

    response = client.post("/ask", json={"question": "Prove that 17 is prime", "tier": "strong"})

    assert response.status_code == 200
    assert response.json()["model"] == "strong-model"
    assert response.json()["tier"] == "strong"
    # The request really went to the strong model's address, with its name
    assert sent[0]["url"].startswith("http://strong.test/v1")
    assert sent[0]["json"]["model"] == "strong-model"


def test_strong_tier_without_api_key_explains_what_to_do(monkeypatch):
    sent = use_fake_model(monkeypatch)  # strong model still has the placeholder key

    response = client.post("/ask", json={"question": "Hello?", "tier": "strong"})

    assert response.status_code == 503
    assert "STRONG_API_KEY" in response.json()["detail"]
    assert sent == []  # the gateway never tried to call the model


def test_strong_request_is_logged_with_its_route(monkeypatch, logged):
    give_strong_model_a_key(monkeypatch)
    use_fake_model(monkeypatch)

    client.post("/ask", json={"question": "Prove that 17 is prime", "tier": "strong"})

    assert logged[0]["route"] == "strong"
    assert logged[0]["model"] == "strong-model"