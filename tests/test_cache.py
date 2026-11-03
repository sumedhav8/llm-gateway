"""Tests for the semantic cache (app/cache.py and how /ask uses it). No real database or Ollama."""

import pytest
from fastapi.testclient import TestClient

from app import cache, config, main, routing
from app.models import ModelReply

client = TestClient(main.app)
VECTOR = [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    """Cache on, fake models, fake embeddings, and catch what would be logged or stored."""
    monkeypatch.setitem(config.MODELS, "cheap",
                        config.ModelConfig("cheap", "http://cheap.test/v1", "key", "claude-haiku-5-5"))
    monkeypatch.setitem(config.MODELS, "strong",
                        config.ModelConfig("strong", "http://strong.test/v1", "key", "claude-sonnet-5-5"))
    monkeypatch.setattr(cache, "CACHE_ENABLED", True)
    monkeypatch.setattr(cache, "CACHE_THRESHOLD", 0.95)
    monkeypatch.setattr(routing, "ROUTING_MODE", "cascade")
    monkeypatch.setattr(main, "embed", lambda texts: [VECTOR for _ in texts])

    state = {"logged": [], "stored": [], "hits": [], "model_calls": 0}
    monkeypatch.setattr(main.db, "log_request", lambda **row: state["logged"].append(row))
    monkeypatch.setattr(cache, "store", lambda *args: state["stored"].append(args))
    monkeypatch.setattr(cache, "record_hit", lambda cache_id: state["hits"].append(cache_id))

    async def fake_ask_model(model, prompt, timeout=120):
        state["model_calls"] += 1
        if prompt.startswith("Here is a question"):
            return ModelReply("95", 200, 2)          # confident self-check
        return ModelReply(f"{model.tier} answer", 20, 50)

    monkeypatch.setattr(routing, "ask_model", fake_ask_model)
    return state


def saved(similarity):
    return cache.CachedAnswer(7, "What's the capital of France?", "Paris", "claude-haiku-5-5", similarity)


def ask(**extra):
    return client.post("/ask", json={"question": "What is France's capital?", "tier": "auto", **extra})


def test_vector_format_for_pgvector():
    assert cache.to_pgvector([0.5, 1, -2.25]) == "[0.5,1,-2.25]"


def test_similar_question_is_answered_from_the_cache(monkeypatch, setup):
    monkeypatch.setattr(cache, "nearest", lambda vector: saved(0.97))

    data = ask().json()

    assert data["answer"] == "Paris"
    assert data["cached"] is True
    assert data["route"] == "cache-hit"
    assert data["cost_usd"] == 0
    assert data["calls"] == 0
    assert data["cached_question"] == "What's the capital of France?"
    assert setup["model_calls"] == 0                 # no model was asked
    assert setup["hits"] == [7]
    assert setup["logged"][0]["route"] == "cache-hit"


def test_not_similar_enough_asks_a_model_and_saves_the_answer(monkeypatch, setup):
    monkeypatch.setattr(cache, "nearest", lambda vector: saved(0.80))

    data = ask().json()

    assert data["cached"] is False
    assert data["route"] == "cascade-accepted"
    assert data["similarity"] == 0.8                 # shown, to help choose the threshold
    assert setup["model_calls"] == 2
    question, vector, answer, model, cost = setup["stored"][0]
    assert (question, vector, answer, model) == ("What is France's capital?", VECTOR, "cheap answer", "claude-haiku-5-5")


def test_empty_cache_is_a_miss(monkeypatch, setup):
    monkeypatch.setattr(cache, "nearest", lambda vector: None)
    data = ask().json()
    assert data["cached"] is False
    assert data["similarity"] is None
    assert len(setup["stored"]) == 1


def test_use_cache_false_skips_the_cache(monkeypatch, setup):
    def should_not_be_called(vector):
        raise AssertionError("cache lookup should be skipped")

    monkeypatch.setattr(cache, "nearest", should_not_be_called)
    data = ask(use_cache=False).json()
    assert data["cached"] is False
    assert setup["stored"] == []


def test_embeddings_down_still_answers(monkeypatch, setup):
    from app.embeddings import EmbeddingError

    def broken_embed(texts):
        raise EmbeddingError("Is Ollama running?")

    monkeypatch.setattr(main, "embed", broken_embed)
    response = ask()
    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert setup["stored"] == []                     # nothing to store without an embedding


def test_router_reuses_the_cache_embedding(monkeypatch):
    class FakeRouter:
        def predict_proba(self, vectors):
            assert vectors == [VECTOR]
            return [[0.1, 0.9]]

    def embed_should_not_run(texts):
        raise AssertionError("the question was embedded twice")

    monkeypatch.setattr(routing, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(routing, "embed", embed_should_not_run)
    assert routing.router_probability("Hi", vector=VECTOR) == 0.9