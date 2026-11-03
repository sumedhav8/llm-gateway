"""Settings shared by every test file (pytest loads this file automatically)."""

import pytest

from app import cache, main
from app.embeddings import EmbeddingError


@pytest.fixture(autouse=True)
def no_real_cache_or_embeddings(monkeypatch):
    """Tests never touch the real cache table or Ollama. Cache tests switch them back on."""
    monkeypatch.setattr(cache, "CACHE_ENABLED", False)

    def no_embeddings(texts):
        raise EmbeddingError("No embeddings in tests")

    monkeypatch.setattr(main, "embed", no_embeddings)