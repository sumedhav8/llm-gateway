import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    """Quick check that the gateway is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start = time.perf_counter()

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
        raise HTTPException(503, "Can't reach the model. Is Ollama running?")

    if response.status_code != 200:
        raise HTTPException(502, f"Model returned an error: {response.text[:300]}")

    data = response.json()
    usage = data.get("usage", {})

    return AskResponse(
        answer=data["choices"][0]["message"]["content"],
        model=MODEL_NAME,
        latency_ms=int((time.perf_counter() - start) * 1000),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )