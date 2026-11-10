"""Tests for RAG part 2: keyword and hybrid search, answers with citations, and their evaluation."""

import pytest
from fastapi.testclient import TestClient

from app import config, main, rag
from app.models import ModelReply
from scripts import eval_hybrid, eval_rag


def hit(source, index, score=0.5, content="text"):
    return rag.Hit(source, index, f"{source} section", content, score)


# --- Keyword and hybrid search -------------------------------------------------

def test_keyword_query_keeps_only_words_once():
    assert rag.keyword_query("How much did the router save on held-out questions? Router!") == \
        "how | much | did | the | router | save | on | held | out | questions"
    assert rag.keyword_query("?!") == ""


def test_rrf_puts_chunks_found_by_both_methods_first():
    vector = [hit("README.md", 1), hit("README.md", 2), hit("DEVLOG.md", 3)]
    keyword = [hit("DEVLOG.md", 3), hit("README.md", 9), hit("README.md", 1)]

    merged = rag.rrf_merge(vector, keyword, k=3)

    assert [(h.source, h.index) for h in merged][:2] == [("README.md", 1), ("DEVLOG.md", 3)]
    assert merged[0].similarity == pytest.approx(1 / 61 + 1 / 63)


def test_retrieve_picks_the_method(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda q, k=4: [hit("v.md", 0)])
    monkeypatch.setattr(rag, "keyword_search", lambda q, k=20: [hit("k.md", 0)])
    assert rag.retrieve("q", method="vector")[0].source == "v.md"
    assert rag.retrieve("q", method="keyword")[0].source == "k.md"
    assert {h.source for h in rag.retrieve("q", k=2, method="hybrid")} == {"v.md", "k.md"}


# --- Answers with citations ----------------------------------------------------

def test_prompt_numbers_the_sources():
    prompt = rag.build_prompt("How much did it save?", [hit("README.md", 1, content="It saved 64%.")])
    assert "[1] (README.md > README.md section)\nIt saved 64%." in prompt
    assert prompt.endswith("Question: How much did it save?\nAnswer:")


def test_cited_numbers():
    assert rag.cited_numbers("It saved 64% [2]. Quality matched [1][2]. See [7].", n_sources=4) == [2, 1]
    assert rag.cited_numbers("No citations here.", n_sources=4) == []


def test_said_not_found():
    assert rag.said_not_found("I don’t know based on the project documents.")
    assert not rag.said_not_found("The router saved 64% [1].")


def ask_rag(monkeypatch, reply_text, hits=None):
    monkeypatch.setitem(config.MODELS, "cheap",
                        config.ModelConfig("cheap", "http://cheap.test/v1", "key", "claude-haiku-5-5"))
    logged = []
    monkeypatch.setattr(main.db, "log_request", lambda **row: logged.append(row))
    monkeypatch.setattr(rag, "retrieve", lambda q, k, method: hits if hits is not None else
                        [hit("README.md", 4, 0.03, "Held-out test: saved 64%"), hit("DEVLOG.md", 2, 0.02)])

    async def fake_ask_model(model, prompt, timeout=120):
        assert "[1] (README.md" in prompt
        return ModelReply(reply_text, 900, 30)

    monkeypatch.setattr(rag, "ask_model", fake_ask_model)
    return TestClient(main.app).post("/rag/ask", json={"question": "How much did the router save?"}), logged


def test_rag_endpoint_returns_answer_and_cited_sources(monkeypatch):
    response, logged = ask_rag(monkeypatch, "On held-out questions it saved 64% [1].")

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "On held-out questions it saved 64% [1]."
    assert data["not_found"] is False
    assert [s["cited"] for s in data["sources"]] == [True, False]
    assert data["sources"][0]["source"] == "README.md"
    assert data["cost_usd"] > 0
    assert logged[0]["route"] == "rag-hybrid"


def test_rag_endpoint_reports_not_found(monkeypatch):
    response, _ = ask_rag(monkeypatch, "I don't know based on the project documents.")
    assert response.json()["not_found"] is True


def test_rag_endpoint_explains_a_missing_database(monkeypatch):
    def broken(q, k, method):
        raise rag.RagError("Search failed. Is the database running?")

    monkeypatch.setattr(rag, "retrieve", broken)
    response = TestClient(main.app).post("/rag/ask", json={"question": "Hi"})
    assert response.status_code == 503
    assert "database" in response.json()["detail"]


# --- Evaluation helpers --------------------------------------------------------

def test_rank_of():
    hits = [hit("a.md", 0), hit("b.md", 3)]
    assert eval_hybrid.rank_of(hits, "b.md", 3) == 2
    assert eval_hybrid.rank_of(hits, "c.md", 0) == eval_hybrid.DEPTH + 1


def test_parse_grades():
    assert eval_rag.parse_grades("CORRECT: YES\nGROUNDED: NO\nREASON: Adds a number not in the sources.") == \
        (True, False, "Adds a number not in the sources.")
    assert eval_rag.parse_grades("**CORRECT:** no\nGROUNDED: yes") == (False, True, "")
    assert eval_rag.parse_grades("I think it's fine") == (None, None, "")


def test_rag_summary():
    rows = [
        {"correct": True, "grounded": True, "not_found": False, "cited": [1], "retrieved_own_chunk": True},
        {"correct": False, "grounded": True, "not_found": True, "cited": [], "retrieved_own_chunk": False},
        {"correct": None, "grounded": None, "not_found": False, "cited": [2], "retrieved_own_chunk": True},
    ]
    s = eval_rag.summarize(rows)
    assert s["graded"] == 2
    assert s["correct"] == 0.5 and s["grounded"] == 1.0
    assert s["not_found"] == pytest.approx(1 / 3)
    assert s["with_citation"] == pytest.approx(2 / 3)