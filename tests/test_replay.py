import pytest

from scripts import replay

LABELS = {
    "m1": {"id": "m1", "method": "number", "reference": "12", "good_enough": True, "strong_correct": True},
    "e1": {"id": "e1", "method": "judge", "reference": "Tokyo", "good_enough": True, "strong_correct": None},
    "e2": {"id": "e2", "method": "judge", "reference": "A poem", "good_enough": False, "strong_correct": None},
}


def row(qid, tier, answer="ok", cost=0.001, latency=1000, calls=2, route="cascade-accepted"):
    return {"id": qid, "status_code": 200, "tier": tier, "answer": answer, "cost_usd": cost,
            "latency_ms": latency, "calls": calls, "route": route, "mode": "cascade"}


def test_math_is_checked_on_the_live_answer():
    assert replay.is_good(row("m1", "cheap", answer="So the total is 12."), LABELS["m1"])
    assert not replay.is_good(row("m1", "cheap", answer="So the total is 13."), LABELS["m1"])


def test_everyday_questions_use_the_labels():
    assert replay.is_good(row("e1", "cheap"), LABELS["e1"])          # Haiku was good enough
    assert not replay.is_good(row("e2", "cheap"), LABELS["e2"])      # Haiku wasn't
    assert replay.is_good(row("e2", "strong"), LABELS["e2"])         # Sonnet assumed correct


def test_summary_counts_cost_quality_and_errors():
    rows = [
        row("m1", "cheap", answer="12", cost=0.001, latency=1000),
        row("e1", "cheap", cost=0.001, latency=2000),
        row("e2", "strong", cost=0.010, latency=9000, calls=3, route="cascade-escalated"),
        {"id": "e2", "status_code": 502, "error": "boom", "mode": "cascade"},
    ]
    strong_cost = {"m1": 0.01, "e1": 0.01, "e2": 0.01}

    summary = replay.summarize(rows, LABELS, strong_cost)

    assert summary["answered"] == 3
    assert summary["errors"] == 1
    assert summary["quality"] == 1.0
    assert summary["cost"] == pytest.approx(0.012)
    assert summary["savings"] == pytest.approx(1 - 0.012 / 0.03)
    assert summary["routes"]["cascade-accepted"] == 2
    assert summary["avg_calls"] == pytest.approx(7 / 3)