import json

import httpx
import pytest

from app import embeddings
from scripts import baseline_routers, routing_data


def make_items():
    # Two questions the cheap model handled, two it didn't.
    return [
        {"id": "a", "category": "fact", "question": "Capital of Japan?", "good_enough": True, "strong_ok": True, "strong_cost": 0.01},
        {"id": "b", "category": "fact", "question": "Capital of France?", "good_enough": True, "strong_ok": True, "strong_cost": 0.01},
        {"id": "c", "category": "math", "question": "Hard problem", "good_enough": False, "strong_ok": True, "strong_cost": 0.03},
        {"id": "d", "category": "math", "question": "Harder problem", "good_enough": False, "strong_ok": False, "strong_cost": 0.05},
    ]


# --- Scoring routing decisions (router day 1) --------------------------------

def test_always_strong_costs_everything():
    result = routing_data.evaluate(make_items(), [False, False, False, False])
    assert result["share_cheap"] == 0
    assert result["quality"] == 0.75          # the strong model got 3 of 4 right
    assert result["cost"] == pytest.approx(0.10)
    assert result["savings"] == pytest.approx(0.0)


def test_always_cheap_is_free_but_worse():
    result = routing_data.evaluate(make_items(), [True, True, True, True])
    assert result["cost"] == 0
    assert result["savings"] == pytest.approx(1.0)
    assert result["quality"] == 0.5


def test_perfect_routing_saves_money_without_losing_quality():
    result = routing_data.evaluate(make_items(), [True, True, False, False])
    assert result["quality"] == 0.75          # same as always strong
    assert result["savings"] == pytest.approx(0.2)


def test_rules_send_simple_questions_to_cheap():
    assert baseline_routers.rules_send_to_cheap("What is the capital of Japan?")
    assert not baseline_routers.rules_send_to_cheap("Sara has 3 apples and buys 5 more. How many now?")
    assert not baseline_routers.rules_send_to_cheap("Write a poem about autumn")


# --- Embeddings (router day 2) -------------------------------------------------

def test_embed_returns_one_vector_per_text_in_order(monkeypatch):
    def fake_embedder(request):
        body = json.loads(request.content)
        assert all(text.startswith(embeddings.EMBED_PREFIX) for text in body["input"])
        # Return them out of order to check the code sorts by index
        data = [{"index": i, "embedding": [float(i), 0.5]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    real_client = httpx.Client
    monkeypatch.setattr(embeddings.httpx, "Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(fake_embedder), **kwargs))

    vectors = embeddings.embed(["one", "two", "three"], batch_size=2)

    assert vectors == [[0.0, 0.5], [1.0, 0.5], [0.0, 0.5]]


def test_embed_explains_when_ollama_is_off(monkeypatch):
    monkeypatch.setattr(embeddings, "EMBED_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(embeddings.EmbeddingError, match="Is Ollama running"):
        embeddings.embed(["hello"])