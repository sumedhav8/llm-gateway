"""Tests for RAG part 1: chunking, embedding prefixes, search and retrieval scores. No real database or Ollama."""

import httpx
import pytest

from app import embeddings, rag
from scripts import eval_retrieval

DOC = """# Triage

Intro paragraph about the gateway.

## Router

The router picks a model.

It saved 64% on held-out questions.

```python
def route(question):

    return "cheap"
```

### Thresholds

Threshold 0.95 was chosen.

## Cache

The cache has two layers.
"""


def test_chunks_follow_the_headings():
    chunks = rag.chunk_markdown(DOC, "README.md", limit=1000)

    assert [c.heading for c in chunks] == ["Triage", "Triage > Router", "Triage > Router > Thresholds", "Triage > Cache"]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[1].content.startswith("Triage > Router\n\n")      # heading kept with the text
    assert "It saved 64%" in chunks[1].content


def test_code_blocks_are_not_cut_and_dont_start_sections():
    chunks = rag.chunk_markdown(DOC, "README.md", limit=1000)
    router = chunks[1].content
    assert 'def route(question):\n\n    return "cheap"\n```' in router   # blank line inside code kept


def test_small_limit_makes_more_chunks_each_within_the_limit():
    text = "## Notes\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = rag.chunk_markdown(text, "notes.md", limit=300)
    assert len(chunks) > 3
    assert all(len(c.content) <= 300 + len("Notes\n\n") for c in chunks)


def test_very_long_paragraph_is_split():
    text = "## Long\n\n" + "x" * 2500
    chunks = rag.chunk_markdown(text, "long.md", limit=1000)
    assert len(chunks) == 3


def test_embed_uses_the_prefix_it_is_given(monkeypatch):
    seen = []

    def fake(request):
        import json
        seen.extend(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    real_client = httpx.Client
    monkeypatch.setattr(embeddings.httpx, "Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs))
    embeddings.embed(["hello"], prefix="search_query: ")
    embeddings.embed(["hello"])
    assert seen == ["search_query: hello", embeddings.EMBED_PREFIX + "hello"]


class FakeConnection:
    def __init__(self, rows, calls):
        self.rows, self.calls = rows, calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        self.calls.append((sql, params))
        return self

    def fetchall(self):
        return self.rows


def test_search_uses_the_query_prefix_and_returns_hits(monkeypatch):
    prefixes, calls = [], []

    def fake_embed(texts, prefix=None):
        prefixes.append(prefix)
        return [[0.1, 0.2]]

    rows = [("README.md", 3, "Triage > Router", "Triage > Router\n\nIt saved 64%", 0.91)]
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag.psycopg, "connect", lambda *a, **k: FakeConnection(rows, calls))

    hits = rag.search("How much did the router save?", k=2)

    assert prefixes == [rag.QUERY_PREFIX]
    assert hits[0].source == "README.md" and hits[0].similarity == pytest.approx(0.91)
    assert calls[0][1][-1] == 2                       # LIMIT k


def test_ranks_recall_and_mrr():
    chunks = [[1, 0], [0, 1], [1, 1]]
    questions = [[1, 0.1], [0.1, 1], [1, 0]]          # third question's own chunk is [1, 1]
    rank_list = eval_retrieval.ranks(questions, chunks, expected=[0, 1, 2])
    assert rank_list == [1, 1, 2]
    assert eval_retrieval.recall_at(rank_list, 1) == pytest.approx(2 / 3)
    assert eval_retrieval.recall_at(rank_list, 3) == 1.0
    assert eval_retrieval.mrr(rank_list) == pytest.approx((1 + 1 + 0.5) / 3)