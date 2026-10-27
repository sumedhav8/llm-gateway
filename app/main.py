import asyncio
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config, db, pricing
from app.models import ModelError, ask_model

app = FastAPI(title="LLM Gateway")


# --- What a request and a response look like ----------------------------
class AskRequest(BaseModel):
    question: str
    tier: Literal["cheap", "strong"] = "cheap"


class AskResponse(BaseModel):
    answer: str
    model: str
    tier: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None means the price is unknown


# --- Helpers --------------------------------------------------------------
def elapsed_ms(start):
    return int((time.perf_counter() - start) * 1000)


async def record(question, model, status_code, *, answer=None, latency_ms=None,
                 input_tokens=None, output_tokens=None, cost_usd=None, error=None):
    """Save one row to the requests table without blocking other requests."""
    await asyncio.to_thread(
        db.log_request,
        question=question,
        answer=answer,
        model=model.name,
        route=model.tier,
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


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
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