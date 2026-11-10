"""Tests for cache safety: questions that must not be cached, expiry, and startup warm-up."""

import pytest
from fastapi.testclient import TestClient

from app import cache, config, main, routing
from app.models import ModelReply


@pytest.mark.parametrize("question, reason", [
    ("Are you going for a walk tomorrow?", "time-sensitive"),
    ("What's the weather like today?", "time-sensitive"),
    ("Who won the game last night?", "time-sensitive"),
    ("What is the latest iPhone?", "time-sensitive"),
    ("What is my name?", "personal"),
    ("Did you read the file I sent?", "personal"),
    ("What is the capital of France?", None),
    ("How do I boil an egg?", None),
    ("Sara has 3 apples and buys 5 more. How many does she have?", None),
    ("Anakin went to the beach today. He caught 10 starfish and 6 sea horses. How many in all?", None),
    ("A bag has a 5% discount. If it is marked $140, how much will you pay?", None),
    ("Last year there were 50 students. This year there was a 20% increase. How many now?", None),
])
def test_uncacheable_reason(question, reason):
    assert cache.uncacheable_reason(question) == reason


def test_time_sensitive_question_skips_the_cache(monkeypatch):
    monkeypatch.setitem(config.MODELS, "cheap",
                        config.ModelConfig("cheap", "http://cheap.test/v1", "key", "claude-haiku-5-5"))
    monkeypatch.setitem(config.MODELS, "strong",
                        config.ModelConfig("strong", "http://strong.test/v1", "key", "claude-sonnet-5-5"))
    monkeypatch.setattr(cache, "CACHE_ENABLED", True)
    monkeypatch.setattr(routing, "ROUTING_MODE", "cascade")
    stored = []

    def should_not_be_called(*args):
        raise AssertionError("the cache should not be used for this question")

    async def fake_ask_model(model, prompt, timeout=120):
        if prompt.startswith("Here is a question"):
            return ModelReply("95", 200, 2)
        return ModelReply("Yes, at 6 pm.", 20, 10)

    monkeypatch.setattr(cache, "exact_match", should_not_be_called)
    monkeypatch.setattr(cache, "nearest", should_not_be_called)
    monkeypatch.setattr(cache, "store", lambda *args: stored.append(args))
    monkeypatch.setattr(main.db, "log_request", lambda **row: None)
    monkeypatch.setattr(routing, "ask_model", fake_ask_model)

    response = TestClient(main.app).post("/ask", json={"question": "Are you going for a walk tomorrow?",
                                                       "tier": "auto"})

    assert response.status_code == 200
    assert response.json()["cache_skipped"] == "time-sensitive"
    assert response.json()["cached"] is False
    assert stored == []                              # not saved either


class FakeConnection:
    """Stands in for a database connection and remembers the last query."""
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return None


def test_lookups_ignore_expired_answers(monkeypatch):
    calls = []
    monkeypatch.setattr(cache.psycopg, "connect", lambda *args, **kwargs: FakeConnection(calls))
    monkeypatch.setattr(cache, "CACHE_TTL_HOURS", 24)

    assert cache.exact_match("What is the capital of France?") is None
    assert cache.nearest([0.1, 0.2]) is None

    for sql, params in calls:
        assert "created_at >= NOW() - make_interval(hours => %s)" in sql
        assert 24 in params


def test_warm_up_loads_embeddings_and_router(monkeypatch):
    loaded = []
    monkeypatch.setattr(main, "embed", lambda texts: loaded.append("embeddings") or [[0.0]])
    monkeypatch.setattr(routing, "load_router", lambda: loaded.append("router"))
    main.warm_up()
    assert loaded == ["embeddings", "router"]


def test_warm_up_never_crashes(monkeypatch):
    def no_router():
        raise routing.RouterUnavailable("no model file")

    # conftest.py already makes embeddings fail; the router fails too
    monkeypatch.setattr(routing, "load_router", no_router)
    main.warm_up()                                   # logs warnings, raises nothing