import asyncio
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config, db, pricing, routing
from app.models import ModelError, ask_model

app = FastAPI(title="LLM Gateway")


# --- What a request and a response look like ----------------------------
class AskRequest(BaseModel):
    question: str
    # cheap / strong: always use that model.  auto: let the gateway decide (see app/routing.py)
    tier: Literal["cheap", "strong", "auto"] = "cheap"
    # Only for tier auto: override ROUTING_MODE from .env for this one request
    mode: Literal["cascade", "router"] | None = None


class AskResponse(BaseModel):
    answer: str
    model: str
    tier: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None means the price is unknown
    route: str = "direct"   # how the model was chosen, e.g. "cascade-accepted"
    confidence: int | None = None            # cascade: the cheap model's self-check score
    router_probability: float | None = None  # router: chance the cheap model is good enough
    calls: int = 1                           # model calls made for this request


# --- Helpers --------------------------------------------------------------
def elapsed_ms(start):
    return int((time.perf_counter() - start) * 1000)


async def record(question, model, status_code, *, answer=None, latency_ms=None,
                 input_tokens=None, output_tokens=None, cost_usd=None, error=None, route=None):
    """Save one row to the requests table without blocking other requests."""
    await asyncio.to_thread(
        db.log_request,
        question=question,
        answer=answer,
        model=model.name,
        route=route or model.tier,
        status_code=status_code,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        error=error,
    )


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    """Quick check that the gateway is running."""
    return {"status": "ok"}


async def ask_auto(req: AskRequest):
    start = time.perf_counter()
    try:
        result = await routing.answer_auto(req.question, mode=req.mode)
    except ModelError as exc:
        await record(req.question, config.MODELS["strong"], exc.status_code, route="auto-failed",
                     latency_ms=elapsed_ms(start), error=exc.message)
        raise HTTPException(exc.status_code, exc.message)

    latency = elapsed_ms(start)
    cost = result.cost_usd
    await record(
        req.question,
        result.model,
        200,
        answer=result.answer,
        latency_ms=latency,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost,
        route=result.route,
    )
    return AskResponse(
        answer=result.answer,
        model=result.model.name,
        tier=result.model.tier,
        latency_ms=latency,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=float(cost) if cost is not None else None,
        route=result.route,
        confidence=result.confidence,
        router_probability=result.router_probability,
        calls=len(result.steps),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if req.tier == "auto":
        return await ask_auto(req)

    model = config.MODELS[req.tier]
    start = time.perf_counter()

    try:
        reply = await ask_model(model, req.question)
    except ModelError as exc:
        await record(req.question, model, exc.status_code,
                     latency_ms=elapsed_ms(start), error=exc.message)
        raise HTTPException(exc.status_code, exc.message)

    latency = elapsed_ms(start)
    cost = pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens)

    await record(
        req.question,
        model,
        200,
        answer=reply.answer,
        latency_ms=latency,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        cost_usd=cost,
    )
    return AskResponse(
        answer=reply.answer,
        model=model.name,
        tier=model.tier,
        latency_ms=latency,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        cost_usd=float(cost) if cost is not None else None,
    )