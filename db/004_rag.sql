-- RAG: document chunks with their embeddings (see app/rag.py).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source       TEXT NOT NULL,              -- file the chunk came from, e.g. README.md
    chunk_index  INTEGER NOT NULL,           -- its position in that file
    heading      TEXT NOT NULL,              -- the section it belongs to
    content      TEXT NOT NULL,              -- heading + text, as embedded
    embedding    vector(768) NOT NULL,
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops);