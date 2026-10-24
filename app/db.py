"""Saving gateway requests to Postgres."""

import logging
import os

import psycopg

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql://gateway:gateway_dev_password@localhost:5432/gateway"

INSERT_SQL = """
    INSERT INTO requests
        (question, answer, model, route, status_code,
         latency_ms, input_tokens, output_tokens, cost_usd, error)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def log_request(
    *,
    question,
    answer,
    model,
    status_code,
    latency_ms=None,
    input_tokens=None,
    output_tokens=None,
    cost_usd=None,
    error=None,
    route="direct",
):
    """Save one request as a row. Never crashes the gateway if the database is down."""
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute(
                INSERT_SQL,
                (question, answer, model, route, status_code,
                 latency_ms, input_tokens, output_tokens, cost_usd, error),
            )
    except psycopg.Error as exc:
        logger.warning("Could not save request to the database: %s", exc)