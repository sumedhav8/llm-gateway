-- Keyword search for RAG (see keyword_search in app/rag.py).
-- tsv is the chunk's words in Postgres full-text form (stemmed, common words removed);
-- Postgres keeps it up to date automatically whenever content changes.

ALTER TABLE rag_chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN index: finds the chunks containing given words quickly.
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx ON rag_chunks USING gin (tsv);