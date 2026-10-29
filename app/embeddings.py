"""Turn text into embeddings: lists of numbers that capture its meaning.

Uses a free embedding model running in Ollama (nomic-embed-text by default),
through the standard /embeddings API.
"""

import os

import httpx

from app import config  # noqa: F401  (importing config loads the .env file)

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text works best with a short task prefix in front of each text
EMBED_PREFIX = os.getenv("EMBED_PREFIX", "classification: ")


class EmbeddingError(Exception):
    pass


def embed(texts, batch_size=32, timeout=120):
    """Return one embedding (a list of floats) per text, in the same order."""
    vectors = []
    with httpx.Client(timeout=timeout) as client:
        for start in range(0, len(texts), batch_size):
            batch = [EMBED_PREFIX + text for text in texts[start:start + batch_size]]
            try:
                response = client.post(
                    f"{EMBED_BASE_URL}/embeddings",
                    json={"model": EMBED_MODEL, "input": batch},
                )
            except httpx.ConnectError:
                raise EmbeddingError("Can't reach the embedding model. Is Ollama running?")
            if response.status_code != 200:
                raise EmbeddingError(f"Embedding model returned an error: {response.text[:300]}")
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in items)
    return vectors