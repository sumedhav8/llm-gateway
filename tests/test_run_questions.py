import asyncio
import json

import httpx

from app import config, models
from scripts import run_questions

CHEAP = config.ModelConfig("cheap", "http://cheap.test/v1", "ollama", "cheap-model")


def test_load_questions_reads_each_line(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "question": "One?"}) + "\n\n"
        + json.dumps({"id": "q2", "question": "Two?"}) + "\n",
        encoding="utf-8",
    )

    questions = run_questions.load_questions(path)

    assert [q["id"] for q in questions] == ["q1", "q2"]
    assert len(run_questions.load_questions(path, limit=1)) == 1


def test_run_one_records_a_successful_answer(monkeypatch):
    def fake_model(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Tokyo"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(models.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(fake_model), **kwargs))

    item = {"id": "fact-01", "category": "fact", "question": "Capital of Japan?", "reference": "Tokyo"}
    row = asyncio.run(run_questions.run_one(CHEAP, item))

    assert row["status_code"] == 200
    assert row["answer"] == "Tokyo"
    assert row["tier"] == "cheap"
    assert row["reference"] == "Tokyo"
    assert row["input_tokens"] == 10


def test_run_one_records_an_error_instead_of_crashing():
    unconfigured = config.ModelConfig("strong", "http://strong.test/v1", config.PLACEHOLDER_KEY, "strong-model")
    item = {"id": "fact-01", "question": "Capital of Japan?"}

    row = asyncio.run(run_questions.run_one(unconfigured, item))

    assert row["status_code"] == 503
    assert row["answer"] is None
    assert "STRONG_API_KEY" in row["error"]