"""Semantic cache: reuse a saved answer when a new question means the same as an old one.

Each answered question is stored with its embedding (a list of numbers that captures its
meaning) in the cache table, using pgvector. For a new question, the gateway finds the
most similar stored question; if the similarity is at least CACHE_THRESHOLD, it returns
the saved answer instantly, without calling any model.

Similarity is cosine similarity: 1.0 means the same meaning, lower means less alike.

The cache must never break the gateway: if the database is down, every lookup is a miss
and nothing is saved, but questions are still answered.
"""

import logging
import os
from dataclasses import dataclass

import psycopg

from app import config  # noqa: F401  (importing config loads the .env file)

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway_dev_password@localhost:5432/gateway")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.95"))


@dataclass
class CachedAnswer:
    id: int
    question: str
    answer: str
    model: str
    similarity: float


def to_pgvector(vector):
    """pgvector reads vectors written like '[0.1,0.2,0.3]'."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def nearest(vector):
    """The most similar stored question, or None if the cache is empty or unreachable."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            row = conn.execute(
                """
                SELECT id, question, answer, model, 1 - (embedding <=> %s::vector) AS similarity
                FROM cache
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (to_pgvector(vector), to_pgvector(vector)),
            ).fetchone()
    except psycopg.Error as exc:
        log.warning("Cache lookup failed, treating as a miss: %s", exc)
        return None
    return CachedAnswer(row[0], row[1], row[2], row[3], float(row[4])) if row else None


def store(question, vector, answer, model, cost_usd):
    """Save an answered question so similar questions can reuse it."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            conn.execute(
                "INSERT INTO cache (question, embedding, answer, model, cost_usd) "
                "VALUES (%s, %s::vector, %s, %s, %s)",
                (question, to_pgvector(vector), answer, model, cost_usd),
            )
    except psycopg.Error as exc:
        log.warning("Could not save to the cache: %s", exc)


def record_hit(cache_id):
    """Count how often each saved answer is reused."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("UPDATE cache SET hits = hits + 1, last_hit_at = NOW() WHERE id = %s", (cache_id,))
    except psycopg.Error as exc:
        log.warning("Could not record a cache hit: %s", exc)
