import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import db

# --- Settings (read from the .env file) ---------------------------------
load_dotenv()
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

app = FastAPI(title="LLM Gateway")


# --- What a request and a response look like ----------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


# --- Logging helper -------------------------------------------------------
async def record(question, status_code, *, answer=None, latency_ms=None,
                 input_tokens=None, output_tokens=None, error=None):
    """Save one row to the requests table without blocking other requests."""
    await asyncio.to_thread(
        db.log_request,
        question=question,
        answer=answer,
        model=MODEL_NAME,
        status_code=status_code,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    """Quick check that the gateway is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start = time.perf_counter()

    def elapsed_ms():
        return int((time.perf_counter() - start) * 1000)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{MODEL_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {MODEL_API_KEY}"},
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": req.question}],
                },
            )
    except httpx.ConnectError:
        await record(req.question, 503, latency_ms=elapsed_ms(), error="Model unreachable")
        raise HTTPException(503, "Can't reach the model. Is Ollama running?")

    if response.status_code != 200:
        detail = f"Model returned an error: {response.text[:300]}"
        await record(req.question, 502, latency_ms=elapsed_ms(), error=detail)
        raise HTTPException(502, detail)

    data = response.json()
    usage = data.get("usage", {})

    result = AskResponse(
        answer=data["choices"][0]["message"]["content"],
        model=MODEL_NAME,
        latency_ms=elapsed_ms(),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
    await record(
        req.question,
        200,
        answer=result.answer,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    return result