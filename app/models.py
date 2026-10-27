"""Talking to AI models using the standard chat-completions format."""

from dataclasses import dataclass

import httpx

from app.config import MAX_TOKENS, ModelConfig


class ModelError(Exception):
    """A model call failed. Carries the HTTP status the gateway should return."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class ModelReply:
    answer: str
    input_tokens: int
    output_tokens: int


async def ask_model(model: ModelConfig, question: str, timeout: float = 120) -> ModelReply:
    """Send one question to one model and return its answer and token counts."""
    if not model.is_configured:
        raise ModelError(
            503,
            f"The {model.tier} model isn't set up yet. "
            f"Add {model.tier.upper()}_API_KEY to your .env file.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{model.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {model.api_key}"},
                json={
                    "model": model.name,
                    "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": question}],
                },
            )
    except httpx.ConnectError:
        raise ModelError(503, f"Can't reach the {model.tier} model. Is it running?")
    except httpx.TimeoutException:
        raise ModelError(504, "The model took too long to answer. Try a smaller model or a shorter question.")

    if response.status_code != 200:
        raise ModelError(502, f"Model returned an error: {response.text[:300]}")

    data = response.json()
    usage = data.get("usage", {})
    return ModelReply(
        answer=data["choices"][0]["message"]["content"],
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )