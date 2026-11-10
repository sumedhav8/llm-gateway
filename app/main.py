import asyncio
import time
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import cache, config, db, pricing, rag, routing
from app.embeddings import EmbeddingError, embed
from app.models import ModelError, ask_model

log = logging.getLogger(__name__)
WARMUP = os.getenv("WARMUP", "true").lower() == "true"


def warm_up():
    """Load the embedding model and the router once at startup, so the first user doesn't wait.

    Without this, the first request took about 24 s (Ollama loading nomic-embed-text, and the
    gateway loading the router). Failures are only logged: the gateway works without them.
    """
    try:
        embed(["warm up"])
    except EmbeddingError as exc:
        log.warning("Warm-up: embeddings unavailable (%s)", exc)
    try:
        routing.load_router()
    except routing.RouterUnavailable as exc:
        log.warning("Warm-up: router unavailable (%s)", exc)


@asynccontextmanager
async def lifespan(app):
    if WARMUP:
        # In the background, so the gateway starts accepting requests right away
        app.state.warmup = asyncio.create_task(asyncio.to_thread(warm_up))
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


# --- What a request and a response look like ----------------------------
class AskRequest(BaseModel):
    question: str
    # cheap / strong: always use that model.  auto: let the gateway decide (see app/routing.py)
    tier: Literal["cheap", "strong", "auto"] = "cheap"
    # Only for tier auto: override ROUTING_MODE from .env for this one request
    mode: Literal["cascade", "router"] | None = None
    # Only for tier auto: set to false to skip the semantic cache for this request
    use_cache: bool = True


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
    cached: bool = False                     # True when the answer came from the semantic cache
    similarity: float | None = None          # similarity to the closest saved question (0 to 1)
    cached_question: str | None = None       # that closest saved question
    cache_skipped: str | None = None         # why the cache wasn't used, e.g. "time-sensitive"


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


# Routes whose answer we don't want to reuse: the cheap answer was kept only because
# the strong model was down, so it may be wrong.
DONT_CACHE_ROUTES = {"cascade-escalation-failed", "router-unavailable-cascade-escalation-failed"}


async def question_vector(question):
    """The question's embedding, or None if the embedding model can't be reached."""
    try:
        return (await asyncio.to_thread(embed, [question]))[0]
    except EmbeddingError:
        return None

def cached_response(hit, route, latency, cost, calls):
    return AskResponse(
        answer=hit.answer, model=hit.model, tier="cache", latency_ms=latency,
        input_tokens=0, output_tokens=0, cost_usd=float(cost), route=route, calls=calls,
        cached=True, similarity=round(hit.similarity, 4), cached_question=hit.question,
    )


async def serve_from_cache(req, hit, route, cost, calls, start):
    """Count the hit, log it, and return the saved answer."""
    await asyncio.to_thread(cache.record_hit, hit.id)
    latency = elapsed_ms(start)
    cached_model = config.ModelConfig("cache", "", "", hit.model)
    await record(req.question, cached_model, 200, answer=hit.answer, latency_ms=latency,
                 input_tokens=0, output_tokens=0, cost_usd=cost, route=route)
    return cached_response(hit, route, latency, cost, calls)


async def ask_auto(req: AskRequest):
    start = time.perf_counter()
    use_cache = req.use_cache and cache.CACHE_ENABLED
    mode = req.mode or routing.ROUTING_MODE

    # 0. Never reuse answers to time-sensitive or personal questions (and don't save them).
    skip_reason = cache.uncacheable_reason(req.question) if use_cache else None
    if skip_reason:
        use_cache = False

    # 1. Exact match: the same question asked before. Safe, and needs no embedding.
    if use_cache:
        hit = await asyncio.to_thread(cache.exact_match, req.question)
        if hit:
            return await serve_from_cache(req, hit, "cache-exact", Decimal("0"), 0, start)

    # 2. Embed the question once; it is used by the semantic cache AND the router.
    vector = await question_vector(req.question) if use_cache or mode == "router" else None

    # 3. Semantic match: a similar saved question, served only if the verifier says
    #    both questions have exactly the same answer.
    nearest = None
    verify_cost, verify_calls = Decimal("0"), 0
    if use_cache and vector is not None:
        nearest = await asyncio.to_thread(cache.nearest, vector)
        if nearest and nearest.similarity >= cache.CACHE_THRESHOLD:
            if not cache.CACHE_VERIFY:
                return await serve_from_cache(req, nearest, "cache-similar", Decimal("0"), 0, start)
            same, verify_cost = await cache.verify(nearest.question, req.question)
            verify_calls = 1
            if same:
                return await serve_from_cache(req, nearest, "cache-verified", verify_cost, 1, start)

    # 4. No cache hit: route to a model.
    try:
        result = await routing.answer_auto(req.question, mode=mode, vector=vector)
    except ModelError as exc:
        await record(req.question, config.MODELS["strong"], exc.status_code, route="auto-failed",
                     latency_ms=elapsed_ms(start), error=exc.message)
        raise HTTPException(exc.status_code, exc.message)

    latency = elapsed_ms(start)
    # The request's cost includes the verifier call, if one was made
    cost = result.cost_usd + verify_cost if result.cost_usd is not None else None
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

    # 5. Save the answer so the same or similar questions can reuse it.
    if use_cache and vector is not None and result.route not in DONT_CACHE_ROUTES:
        await asyncio.to_thread(cache.store, req.question, vector, result.answer, result.model.name,
                                result.cost_usd)

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
        calls=len(result.steps) + verify_calls,
        similarity=round(nearest.similarity, 4) if nearest else None,
        cached_question=nearest.question if nearest else None,
        cache_skipped=skip_reason,
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



# --- RAG: answers about the project's own documents -----------------------------
class RagRequest(BaseModel):
    question: str
    k: int = Field(4, ge=1, le=10)                              # how many chunks to retrieve
    method: Literal["hybrid", "vector", "keyword"] = "hybrid"   # how to find them
    tier: Literal["cheap", "strong"] = "cheap"                  # which model writes the answer


class RagSource(BaseModel):
    number: int          # the [n] the answer uses to cite it
    source: str          # file, e.g. README.md
    heading: str         # section in that file
    score: float         # retrieval score (RRF for hybrid, similarity for vector)
    cited: bool          # whether the answer actually cites it
    excerpt: str


class RagResponse(BaseModel):
    answer: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    not_found: bool      # the model said the documents don't contain the answer
    sources: list[RagSource]


@app.post("/rag/ask", response_model=RagResponse)
async def rag_ask(req: RagRequest):
    """Answer a question about this project from its README and DEVLOG, with citations."""
    start = time.perf_counter()
    model = config.MODELS[req.tier]
    try:
        result = await rag.answer(req.question, k=req.k, method=req.method, tier=req.tier)
    except (rag.RagError, EmbeddingError) as exc:
        raise HTTPException(503, str(exc))
    except ModelError as exc:
        await record(req.question, model, exc.status_code, route=f"rag-{req.method}",
                     latency_ms=elapsed_ms(start), error=exc.message)
        raise HTTPException(exc.status_code, exc.message)

    latency = elapsed_ms(start)
    await record(req.question, model, 200, answer=result.answer, latency_ms=latency,
                 input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                 cost_usd=result.cost_usd, route=f"rag-{req.method}")
    return RagResponse(
        answer=result.answer,
        model=model.name,
        latency_ms=latency,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=float(result.cost_usd) if result.cost_usd is not None else None,
        not_found=result.not_found,
        sources=[RagSource(number=n, source=hit.source, heading=hit.heading, score=round(hit.similarity, 4),
                           cited=n in result.cited, excerpt=hit.content[:300])
                 for n, hit in enumerate(result.sources, 1)],
    )