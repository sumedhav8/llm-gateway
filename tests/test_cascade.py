import pytest

from scripts import cascade


def make_items():
    # Two questions the cheap model handled, two it didn't (same as test_routing.py).
    return [
        {"id": "a", "category": "fact", "good_enough": True, "strong_ok": True, "strong_cost": 0.01, "cheap_cost": 0.001},
        {"id": "b", "category": "fact", "good_enough": True, "strong_ok": True, "strong_cost": 0.01, "cheap_cost": 0.001},
        {"id": "c", "category": "math", "good_enough": False, "strong_ok": True, "strong_cost": 0.03, "cheap_cost": 0.001},
        {"id": "d", "category": "math", "good_enough": False, "strong_ok": False, "strong_cost": 0.05, "cheap_cost": 0.001},
    ]


def test_parse_confidence():
    assert cascade.parse_confidence("85") == 85
    assert cascade.parse_confidence("Confidence: 40/100") == 40
    assert cascade.parse_confidence("I'd say 250... no, 90") == 90   # skips numbers above 100
    assert cascade.parse_confidence("Very sure!") is None


def test_numbers_agree():
    assert cascade.numbers_agree("So she has 12 apples.", "The answer is 12")
    assert cascade.numbers_agree("Total: $1,200", "1200")
    assert not cascade.numbers_agree("12", "13")
    assert not cascade.numbers_agree("no number here", "12")


def test_cascade_pays_for_cheap_answer_and_check_on_every_question():
    items = make_items()
    # Accept the two good answers, escalate the two bad ones; each check costs 0.0005
    result = cascade.cascade_evaluate(items, [True, True, False, False], [0.0005] * 4)
    assert result["quality"] == 0.75                      # same as always strong
    assert result["cost"] == pytest.approx(4 * 0.001 + 4 * 0.0005 + 0.03 + 0.05)
    assert result["share_cheap"] == 0.5


def test_escalating_everything_costs_more_than_always_strong():
    items = make_items()
    result = cascade.cascade_evaluate(items, [False] * 4, [0.0005] * 4)
    assert result["cost"] > sum(item["strong_cost"] for item in items)
    assert result["savings"] == 0.0                       # never shows negative savings