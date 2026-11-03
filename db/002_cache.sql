-- Semantic cache: answered questions with their embeddings (see app/cache.py).
-- 768 numbers per embedding, the size nomic-embed-text produces.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cache (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    question     TEXT NOT NULL,
    embedding    vector(768) NOT NULL,
    answer       TEXT NOT NULL,
    model        TEXT NOT NULL,
    cost_usd     NUMERIC(12, 8),          -- what the original answer cost (saved on every hit)
    hits         INTEGER NOT NULL DEFAULT 0,
    last_hit_at  TIMESTAMPTZ
);

-- HNSW index: finds the most similar embedding quickly, even with many rows.
CREATE INDEX IF NOT EXISTS cache_embedding_idx ON cache USING hnsw (embedding vector_cosine_ops);
