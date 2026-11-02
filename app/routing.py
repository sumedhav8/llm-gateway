"""Automatic routing: decide which model answers, for requests with "tier": "auto".

Two modes, chosen with ROUTING_MODE in .env:

  cascade (default)  The cheap model answers, then rates its own answer from 0 to 100.
                     If the rating is below CASCADE_THRESHOLD, the strong model answers instead.
  router             The trained router (data/models/router.joblib) reads the question and
                     predicts whether the cheap model will be good enough. If the probability
                     is at least ROUTER_THRESHOLD, the cheap model answers; otherwise the strong one.

Both modes were compared offline in scripts/cascade.py (see the README).
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from app import config, pricing
from app.embeddings import EmbeddingError, embed
from app.models import ModelError, ask_model

ROUTING_MODE = os.getenv("ROUTING_MODE", "cascade")
CASCADE_THRESHOLD = int(os.getenv("CASCADE_THRESHOLD", "85"))
ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.5"))
ROUTER_MODEL_PATH = Path(os.getenv("ROUTER_MODEL_PATH", "data/models/router.joblib"))

# Used by the live gateway AND by scripts/cascade.py, so offline tests and live traffic match.
SELF_CHECK_PROMPT = """Here is a question and an answer written by an AI assistant.

Question:
{question}

Answer:
{answer}

How confident are you that this answer is correct and complete, so that a user would be satisfied with it?
Reply with only a whole number from 0 to 100."""


def parse_confidence(text):
    """The first whole number from 0 to 100 in the reply, or None."""
    for match in re.findall(r"\d+", text or ""):
        value = int(match)
        if 0 <= value <= 100:
            return value
    return None


class RouterUnavailable(Exception):
    pass


# --- What happened while answering one request -----------------------------------

@dataclass
class Step:
    purpose: str                 # "answer", "self-check" or "escalate"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal | None = Decimal("0")
    error: str | None = None


@dataclass
class RoutedAnswer:
    answer: str
    model: config.ModelConfig    # the model whose answer is returned
    route: str                   # e.g. "cascade-accepted", "cascade-escalated", "router-cheap"
    steps: list = field(default_factory=list)
    confidence: int | None = None
    router_probability: float | None = None

    @property
    def input_tokens(self):
        return sum(step.input_tokens for step in self.steps)

    @property
    def output_tokens(self):
        return sum(step.output_tokens for step in self.steps)

    @property
    def cost_usd(self):
        """Total cost of every call made for this request (None if any price is unknown)."""
        costs = [step.cost_usd for step in self.steps]
        return None if any(cost is None for cost in costs) else sum(costs, Decimal("0"))


async def call(model, prompt, purpose, steps):
    """Ask a model, and record the call (or its failure) as a step."""
    try:
        reply = await ask_model(model, prompt)
    except ModelError as exc:
        steps.append(Step(purpose, model.name, error=exc.message))
        raise
    steps.append(Step(purpose, model.name, reply.input_tokens, reply.output_tokens,
                      pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens)))
    return reply


# --- Cascade -------------------------------------------------------------------------

async def cascade(question, threshold=None):
    threshold = CASCADE_THRESHOLD if threshold is None else threshold
    cheap, strong = config.MODELS["cheap"], config.MODELS["strong"]
    steps = []

    try:
        first = await call(cheap, question, "answer", steps)
    except ModelError:
        # The cheap model is down: go straight to the strong model.
        reply = await call(strong, question, "escalate", steps)
        return RoutedAnswer(reply.answer, strong, "cascade-cheap-failed", steps)

    try:
        check = await call(cheap, SELF_CHECK_PROMPT.format(question=question, answer=first.answer),
                           "self-check", steps)
        confidence = parse_confidence(check.answer)
    except ModelError:
        confidence = None   # no rating, so we can't trust the answer: escalate

    if confidence is not None and confidence >= threshold:
        return RoutedAnswer(first.answer, cheap, "cascade-accepted", steps, confidence=confidence)

    try:
        reply = await call(strong, question, "escalate", steps)
    except ModelError:
        # The strong model is down: the cheap answer is better than nothing.
        return RoutedAnswer(first.answer, cheap, "cascade-escalation-failed", steps, confidence=confidence)
    return RoutedAnswer(reply.answer, strong, "cascade-escalated", steps, confidence=confidence)


# --- Router --------------------------------------------------------------------------

_router_model = None


def load_router():
    global _router_model
    if _router_model is None:
        if not ROUTER_MODEL_PATH.exists():
            raise RouterUnavailable(f"No trained router at {ROUTER_MODEL_PATH}. "
                                    "Run: python -m scripts.train_router")
        import joblib
        _router_model = joblib.load(ROUTER_MODEL_PATH)
    return _router_model


def router_probability(question):
    """The router's estimate that the cheap model will be good enough (0 to 1)."""
    model = load_router()
    try:
        vector = embed([question])[0]
    except EmbeddingError as exc:
        raise RouterUnavailable(str(exc))
    return float(model.predict_proba([vector])[0][1])


async def route_with_router(question, threshold=None):
    threshold = ROUTER_THRESHOLD if threshold is None else threshold
    try:
        probability = await asyncio.to_thread(router_probability, question)
    except RouterUnavailable:
        # Without the router we can't predict, so use the cascade instead.
        result = await cascade(question)
        result.route = "router-unavailable-" + result.route
        return result

    tier = "cheap" if probability >= threshold else "strong"
    model = config.MODELS[tier]
    steps = []
    reply = await call(model, question, "answer", steps)
    return RoutedAnswer(reply.answer, model, f"router-{tier}", steps, router_probability=probability)


# --- Entry point ---------------------------------------------------------------------

async def answer_auto(question, mode=None):
    mode = mode or ROUTING_MODE
    if mode == "router":
        return await route_with_router(question)
    return await cascade(question)