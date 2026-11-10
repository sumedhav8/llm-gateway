"""RAG, part 1: split documents into chunks, store their embeddings, and find the right chunks.

RAG (retrieval-augmented generation) answers a question from YOUR documents: first find the
pieces of text that are relevant (retrieval), then give them to a model with the question
(generation, in part 2).

How documents are prepared:
1. Split each markdown file into sections at its headings (#, ##, ...).
2. Pack each section's paragraphs into chunks of up to CHUNK_CHARS characters. Code blocks
   are never cut in the middle.
3. Put the section heading in front of each chunk, so a chunk like "AUC 0.735" still says
   which part of the document it came from.
4. Embed every chunk with "search_document: " in front (questions get "search_query: "),
   and store it in the rag_chunks table (pgvector).
"""

import os
import re
from dataclasses import dataclass

import psycopg

from app import config  # noqa: F401  (importing config loads the .env file)
from app.cache import to_pgvector
from app.embeddings import embed

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway_dev_password@localhost:5432/gateway")
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1000"))
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class RagError(Exception):
    pass


@dataclass
class Chunk:
    source: str      # file it came from, e.g. "README.md"
    index: int       # position in that file: 0, 1, 2...
    heading: str     # the section it belongs to, e.g. "Router v2 > Live replay"
    content: str     # heading line + text, exactly what gets embedded and shown to the model


@dataclass
class Hit:
    source: str
    index: int
    heading: str
    content: str
    similarity: float


# --- 1-3. Splitting documents into chunks --------------------------------------------

def sections(text):
    """Split markdown into (heading path, list of blocks). A block is a paragraph or a code block."""
    path = []                     # headings above the current line: [(level, title), ...]
    blocks, current = [], []
    in_code = False
    result = []

    def end_block():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    def end_section():
        end_block()
        kept = [b for b in blocks if b]
        if kept:
            result.append((" > ".join(title for _, title in path), kept))
        blocks.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            current.append(line)
            if not in_code:
                end_block()                     # a code block is one block
            continue
        heading = None if in_code else HEADING_RE.match(line)
        if heading:
            end_section()
            level = len(heading.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level] + [(level, heading.group(2))]
        elif not in_code and not line.strip():
            end_block()                         # blank line ends a paragraph
        else:
            current.append(line)
    end_section()
    return result


def split_long(block, limit):
    """Cut a block that is longer than the limit at line breaks (or hard, if one line is too long)."""
    pieces, current = [], ""
    for line in block.splitlines():
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        if current and len(current) + 1 + len(line) > limit:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(text, source, limit=None):
    """Turn a markdown document into chunks of at most `limit` characters of text each."""
    limit = limit or CHUNK_CHARS
    chunks = []
    for heading, blocks in sections(text):
        body = ""
        for block in blocks:
            for piece in (split_long(block, limit) if len(block) > limit else [block]):
                if body and len(body) + 2 + len(piece) > limit:
                    chunks.append((heading, body))
                    body = piece
                else:
                    body = f"{body}\n\n{piece}" if body else piece
        if body:
            chunks.append((heading, body))
    return [Chunk(source, i, heading, f"{heading}\n\n{body}" if heading else body)
            for i, (heading, body) in enumerate(chunks)]


# --- 4. Storing and searching ------------------------------------------------------

def index_files(paths, limit=None):
    """Chunk, embed and store each file, replacing what was stored for it before."""
    chunks = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            chunks += chunk_markdown(f.read(), str(path).replace("\\", "/"), limit)
    vectors = embed([c.content for c in chunks], prefix=DOC_PREFIX)
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            for source in {c.source for c in chunks}:
                conn.execute("DELETE FROM rag_chunks WHERE source = %s", (source,))
            for chunk, vector in zip(chunks, vectors):
                conn.execute(
                    "INSERT INTO rag_chunks (source, chunk_index, heading, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (chunk.source, chunk.index, chunk.heading, chunk.content, to_pgvector(vector)),
                )
    except psycopg.Error as exc:
        raise RagError(f"Couldn't save the chunks. Is the database running, and was db/004_rag.sql run? ({exc})")
    return chunks


def search(question, k=4, vector=None):
    """The k chunks most similar to the question, most similar first."""
    if vector is None:
        vector = embed([question], prefix=QUERY_PREFIX)[0]
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            rows = conn.execute(
                """
                SELECT source, chunk_index, heading, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (to_pgvector(vector), to_pgvector(vector), k),
            ).fetchall()
    except psycopg.Error as exc:
        raise RagError(f"Search failed. Is the database running, and was db/004_rag.sql run? ({exc})")
    return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]