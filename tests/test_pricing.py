from decimal import Decimal

from app import pricing


def test_local_model_is_free():
    assert pricing.request_cost("llama3.2", 1000, 500) == 0


def test_paid_model_cost_is_calculated(monkeypatch):
    # A made-up paid model: $3 per million input tokens, $15 per million output tokens
    monkeypatch.setitem(pricing.PRICES_PER_MILLION, "test-model", (Decimal("3"), Decimal("15")))

    cost = pricing.request_cost("test-model", 1000, 500)

    # 1,000 x $3/M = $0.003, plus 500 x $15/M = $0.0075, total $0.0105
    assert cost == Decimal("0.0105")


def test_unknown_model_cost_is_none():
    assert pricing.request_cost("some-unknown-model", 1000, 500) is None