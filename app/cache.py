"""Two-layer cache: reuse a saved answer only when it is really the answer to the new question.

Layer 1, exact match: the same question text (ignoring upper/lower case and extra spaces)
gets the saved answer. Always safe, and needs no embedding.

Layer 2, semantic match with a verifier: find the most similar saved question with
pgvector (cosine similarity). Similarity alone is NOT safe: "Sara has 3 apples" and
"Sara has 4 apples" score about 0.999 but need different answers (see evals/cache_threshold.png).
So when a saved question is similar enough (CACHE_THRESHOLD), the cheap model is asked
whether both questions have exactly the same correct answer. Only a YES serves the saved answer.

The cache must never break the gateway: if the database or a model is unreachable, it
counts as a miss and the question is answered normally.
"""

import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal

import psycopg

from app import config, pricing
from app.models import ModelError, ask_model

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway_dev_password@localhost:5432/gateway")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
# Saved questions at least this similar are sent to the verifier (0.86 kept every paraphrase in the eval)
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.86"))
# Without the verifier, the similarity threshold alone decides (unsafe, kept only for comparison)
CACHE_VERIFY = os.getenv("CACHE_VERIFY", "true").lower() == "true"
# Saved answers older than this are ignored, so facts that change don't stay stale forever
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "168"))   # 7 days

VERIFY_PROMPT = """Two users asked an AI assistant these questions.

Question A:
{a}

Question B:
{b}

Would exactly the same answer be fully correct for both questions? Check every number, name, place, date and condition.
Reply with only YES or NO."""


@dataclass
class CachedAnswer:
    id: int
    question: str
    answer: str
    model: str
    similarity: float


# --- Small helpers -------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text):
    """All the numbers in a question, in order ('1,200' counts as 1200)."""
    return [float(n) for n in NUMBER_RE.findall(text.replace(",", ""))]


def same_numbers(question_a, question_b):
    """True when both questions contain exactly the same numbers.

    Embeddings barely notice a changed number ("3 apples" vs "4 apples"), but the answer
    changes completely, so a cache hit is only safe if the numbers match.
    """
    return sorted(numbers_in(question_a)) == sorted(numbers_in(question_b))

# Questions whose right answer depends on WHEN or WHO is asking. Their answers are never
# reused: "Are you going for a walk tomorrow?" asked today and next week are different questions.
TIME_SENSITIVE_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|now|currently|current|latest|recent|recently|"
    r"this (?:morning|evening|week|weekend|month|year)|next (?:week|month|year)|"
    r"last (?:night|week|month|year)|news|weather|forecast|stock price)\b",
    re.IGNORECASE,
)
PERSONAL_RE = re.compile(r"\b(my|mine|our|ours)\b|\b(?:are|were|will|did|have) you\b", re.IGNORECASE)


def uncacheable_reason(question):
    """Why this question's answer must not be reused ("time-sensitive" or "personal"), or None.

    A question that brings its own numbers (2 or more) is treated as self-contained, like a
    math word problem: there, "today" or "you" are part of the story, not the real calendar
    or the real user, so the answer never changes.
    """
    if len(numbers_in(question)) >= 2:
        return None
    if TIME_SENSITIVE_RE.search(question):
        return "time-sensitive"
    if PERSONAL_RE.search(question):
        return "personal"
    return None


def normalize(question):
    """The form used for exact matching: lower case, single spaces, no spaces at the ends."""
    return " ".join(question.split()).lower()


def to_pgvector(vector):
    """pgvector reads vectors written like '[0.1,0.2,0.3]'."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def parse_yes_no(text):
    """True for YES, False for NO, None if the reply is unclear."""
    for word in re.findall(r"[A-Za-z]+", text or ""):
        if word.upper() == "YES":
            return True
        if word.upper() == "NO":
            return False
    return None


# --- Layer 1: exact match ------------------------------------------------------------

def exact_match(question):
    """A saved answer to exactly this question (after normalize), or None."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT id, question, answer, model FROM cache "
                "WHERE question_norm = %s AND created_at >= NOW() - make_interval(hours => %s) "
                "ORDER BY id DESC LIMIT 1",
                (normalize(question), CACHE_TTL_HOURS),
            ).fetchone()
    except psycopg.Error as exc:
        log.warning("Exact cache lookup failed, treating as a miss: %s", exc)
        return None
    return CachedAnswer(row[0], row[1], row[2], row[3], 1.0) if row else None


# --- Layer 2: semantic match + verifier ----------------------------------------------

def nearest(vector):
    """The most similar stored question, or None if the cache is empty or unreachable."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            row = conn.execute(
                """
                SELECT id, question, answer, model, 1 - (embedding <=> %s::vector) AS similarity
                FROM cache
                WHERE created_at >= NOW() - make_interval(hours => %s)
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (to_pgvector(vector), CACHE_TTL_HOURS, to_pgvector(vector)),
            ).fetchone()
    except psycopg.Error as exc:
        log.warning("Cache lookup failed, treating as a miss: %s", exc)
        return None
    return CachedAnswer(row[0], row[1], row[2], row[3], float(row[4])) if row else None


async def verify(saved_question, new_question, model=None):
    """Ask a model whether both questions have exactly the same answer.

    Returns (same, cost). same is False when the model says NO, gives an unclear reply,
    or can't be reached: when in doubt, don't serve a saved answer.
    """
    model = model or config.MODELS["cheap"]
    try:
        reply = await ask_model(model, VERIFY_PROMPT.format(a=saved_question, b=new_question))
    except ModelError as exc:
        log.warning("Cache verifier failed, treating as a miss: %s", exc.message)
        return False, Decimal("0")
    cost = pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens) or Decimal("0")
    return parse_yes_no(reply.answer) is True, cost


# --- Saving --------------------------------------------------------------------------

def store(question, vector, answer, model, cost_usd):
    """Save an answered question so the same or similar questions can reuse it."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            conn.execute(
                "INSERT INTO cache (question, question_norm, embedding, answer, model, cost_usd) "
                "VALUES (%s, %s, %s::vector, %s, %s, %s)",
                (question, normalize(question), to_pgvector(vector), answer, model, cost_usd),
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