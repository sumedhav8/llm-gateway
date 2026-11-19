"""Tests for the chat page and the price endpoint it uses."""

from fastapi.testclient import TestClient

from app import config, main

client = TestClient(main.app)


def test_chat_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Triage LLM Gateway" in response.text


def test_prices_for_the_strong_model(monkeypatch):
    monkeypatch.setitem(config.MODELS, "strong",
                        config.ModelConfig("strong", "http://strong.test/v1", "key", "claude-sonnet-5-5"))
    data = client.get("/api/prices").json()["strong"]
    assert data == {"model": "claude-sonnet-5-5", "input": 2.0, "output": 10.0}


def test_prices_for_an_unknown_model(monkeypatch):
    monkeypatch.setitem(config.MODELS, "strong",
                        config.ModelConfig("strong", "http://strong.test/v1", "key", "mystery-model"))
    data = client.get("/api/prices").json()["strong"]
    assert data["input"] is None and data["output"] is None