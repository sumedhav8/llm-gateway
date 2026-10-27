"""Cost of a request, calculated from its token counts."""

from decimal import Decimal

# Price in US dollars per 1 million tokens: (input price, output price).
# Local Ollama models run on your own computer, so they cost $0.
# Paid model prices come from the provider's official pricing page.
PRICES_PER_MILLION = {
    "llama3.2": (Decimal("0"), Decimal("0")),
    "llama3.2:1b": (Decimal("0"), Decimal("0")),
    # TODO: confirm on the official Claude pricing page before relying on costs
    "claude-sonnet-5-5": (Decimal("2"), Decimal("10")),
}

ONE_MILLION = Decimal(1_000_000)


def request_cost(model, input_tokens, output_tokens):
    """Return the cost in dollars, or None if the model's price is unknown."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / ONE_MILLION