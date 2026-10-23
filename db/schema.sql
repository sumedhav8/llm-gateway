-- Turn on pgvector (needed later for the semantic cache)
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per request the gateway handles
CREATE TABLE IF NOT EXISTS requests (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    answer          TEXT,
    model           TEXT NOT NULL,
    route           TEXT NOT NULL DEFAULT 'direct',
    status_code     INTEGER NOT NULL,
    latency_ms      INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        NUMERIC(12, 8),
    error           TEXT
);

-- Speed up the queries the dashboard will run most
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests (created_at);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests (model);