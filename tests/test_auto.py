"""Tests for "tier": "auto" (app/routing.py). No real models, database or router are used."""

import asyncio
from decimal import Decimal

import pytest

from app import config, routing
from app.models import ModelError, ModelReply

CHEAP = config.ModelConfig("cheap", "http://cheap.test/v1", "key", "claude-haiku-5-5")
STRONG = config.ModelConfig("strong", "http://strong.test/v1", "key", "claude-sonnet-5-5")


@pytest.fixture(autouse=True)
def test_models(monkeypatch):
    monkeypatch.setitem(config.MODELS, "cheap", CHEAP)
    monkeypatch.setitem(config.MODELS, "strong", STRONG)


def fake_models(monkeypatch, *, self_check="95", cheap_down=False, strong_down=False):
    """Replace ask_model with a fake. Returns the list of (model, prompt) calls it received."""
    calls = []

    async def fake_ask_model(model, prompt, timeout=120):
        calls.append((model.tier, prompt))
        if model.tier == "cheap" and cheap_down:
            raise ModelError(503, "cheap is down")
        if model.tier == "strong" and strong_down:
            raise ModelError(503, "strong is down")
        if model.tier == "cheap" and prompt.startswith("Here is a question"):
            return ModelReply(self_check, 200, 2)
        return ModelReply(f"{model.tier} answer", 20, 50)

    monkeypatch.setattr(routing, "ask_model", fake_ask_model)
    return calls


def run(coro):
    return asyncio.run(coro)


# --- Cascade -----------------------------------------------------------------

def test_confident_cheap_answer_is_kept(monkeypatch):
    calls = fake_models(monkeypatch, self_check="95")
    result = run(routing.cascade("What is 2 + 2?", threshold=85))

    assert result.answer == "cheap answer"
    assert result.route == "cascade-accepted"
    assert result.confidence == 95
    assert [tier for tier, _ in calls] == ["cheap", "cheap"]     # answer + self-check, no Sonnet


def test_unsure_cheap_answer_is_escalated(monkeypatch):
    calls = fake_models(monkeypatch, self_check="Confidence: 40")
    result = run(routing.cascade("Prove the Riemann hypothesis", threshold=85))

    assert result.answer == "strong answer"
    assert result.model.tier == "strong"
    assert result.route == "cascade-escalated"
    assert result.confidence == 40
    assert [tier for tier, _ in calls] == ["cheap", "cheap", "strong"]


def test_unreadable_self_check_is_escalated(monkeypatch):
    fake_models(monkeypatch, self_check="I'm quite sure!")
    result = run(routing.cascade("Hi", threshold=85))
    assert result.route == "cascade-escalated"


def test_cost_adds_up_every_call(monkeypatch):
    fake_models(monkeypatch, self_check="10")
    result = run(routing.cascade("Hi", threshold=85))

    expected = sum((routing.pricing.request_cost(model, i, o) for model, i, o in [
        ("claude-haiku-5-5", 20, 50),     # cheap answer
        ("claude-haiku-5-5", 200, 2),     # self-check
        ("claude-sonnet-5-5", 20, 50),    # escalation
    ]), Decimal("0"))
    assert result.cost_usd == expected
    assert result.input_tokens == 240
    assert len(result.steps) == 3


def test_cheap_model_down_goes_straight_to_strong(monkeypatch):
    fake_models(monkeypatch, cheap_down=True)
    result = run(routing.cascade("Hi"))
    assert result.answer == "strong answer"
    assert result.route == "cascade-cheap-failed"


def test_strong_model_down_returns_cheap_answer(monkeypatch):
    fake_models(monkeypatch, self_check="10", strong_down=True)
    result = run(routing.cascade("Hi", threshold=85))
    assert result.answer == "cheap answer"
    assert result.route == "cascade-escalation-failed"


def test_both_models_down_raises(monkeypatch):
    fake_models(monkeypatch, cheap_down=True, strong_down=True)
    with pytest.raises(ModelError):
        run(routing.cascade("Hi"))


# --- Router ------------------------------------------------------------------

def test_router_sends_easy_question_to_cheap(monkeypatch):
    calls = fake_models(monkeypatch)
    monkeypatch.setattr(routing, "router_probability", lambda question: 0.9)
    result = run(routing.route_with_router("Capital of Japan?", threshold=0.5))

    assert result.route == "router-cheap"
    assert result.router_probability == 0.9
    assert [tier for tier, _ in calls] == ["cheap"]                # one call, no self-check


def test_router_sends_hard_question_to_strong(monkeypatch):
    fake_models(monkeypatch)
    monkeypatch.setattr(routing, "router_probability", lambda question: 0.2)
    result = run(routing.route_with_router("Hard question", threshold=0.5))
    assert result.route == "router-strong"
    assert result.answer == "strong answer"


def test_router_unavailable_falls_back_to_cascade(monkeypatch):
    fake_models(monkeypatch, self_check="95")

    def broken_router(question):
        raise routing.RouterUnavailable("Is Ollama running?")

    monkeypatch.setattr(routing, "router_probability", broken_router)
    result = run(routing.route_with_router("Hi"))
    assert result.route == "router-unavailable-cascade-accepted"


# --- Through the API ---------------------------------------------------------

def test_auto_tier_through_the_api(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    rows = []
    monkeypatch.setattr(main.db, "log_request", lambda **row: rows.append(row))
    fake_models(monkeypatch, self_check="10")
    monkeypatch.setattr(routing, "ROUTING_MODE", "cascade")
    monkeypatch.setattr(routing, "CASCADE_THRESHOLD", 85)

    response = TestClient(main.app).post("/ask", json={"question": "Hi", "tier": "auto"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "strong answer"
    assert data["route"] == "cascade-escalated"
    assert data["confidence"] == 10
    assert data["calls"] == 3
    assert rows[0]["route"] == "cascade-escalated"
    assert rows[0]["model"] == "claude-sonnet-5-5"