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

import asyncio
import os
import re
from dataclasses import dataclass, field

import psycopg

from app import config, pricing
from app.cache import to_pgvector
from app.embeddings import embed
from app.models import ask_model

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



def stored_chunks():
    """Every stored chunk's text, by (source, chunk_index)."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            rows = conn.execute("SELECT source, chunk_index, content FROM rag_chunks").fetchall()
    except psycopg.Error as exc:
        raise RagError(f"Couldn't read the chunks. Is the database running? ({exc})")
    return {(source, index): content for source, index, content in rows}


# --- Part 2a. Keyword search and hybrid search ----------------------------------------
#
# Embeddings capture meaning but can miss exact terms ("held-out", "tier", "79%").
# Keyword search (Postgres full-text search) finds exact terms but misses synonyms.
# Hybrid search runs both and merges the two rankings with Reciprocal Rank Fusion (RRF):
# each chunk scores 1 / (RRF_K + rank) in each list, and the scores are added up.
# A chunk near the top of both lists wins; RRF_K = 60 is the usual value.

RRF_K = 60
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def keyword_query(question):
    """A Postgres OR query of the question's words: 'held | out | router | save'.

    Only letters and digits are kept, so the query is always valid. to_tsquery drops
    common words like 'the' and reduces words to their stem ('saved' -> 'save').
    """
    words = WORD_RE.findall(question.lower())
    return " | ".join(dict.fromkeys(words))     # remove repeats, keep order


def keyword_search(question, k=20):
    """The k chunks whose words best match the question's words (Postgres full-text search)."""
    query = keyword_query(question)
    if not query:
        return []
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            rows = conn.execute(
                """
                SELECT source, chunk_index, heading, content, ts_rank_cd(tsv, query) AS score
                FROM rag_chunks, to_tsquery('english', %s) AS query
                WHERE tsv @@ query
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, k),
            ).fetchall()
    except psycopg.Error as exc:
        raise RagError(f"Keyword search failed. Was db/005_rag_keywords.sql run? ({exc})")
    return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]


def rrf_merge(*rankings, k=4):
    """Merge ranked lists of hits with Reciprocal Rank Fusion; returns the top k."""
    scores, hits = {}, {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            key = (hit.source, hit.index)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            hits.setdefault(key, hit)
    best = sorted(scores, key=lambda key: -scores[key])[:k]
    return [Hit(hits[key].source, hits[key].index, hits[key].heading, hits[key].content, scores[key])
            for key in best]


def hybrid_search(question, k=4, candidates=20):
    """Embedding search and keyword search together, merged with RRF."""
    return rrf_merge(search(question, k=candidates), keyword_search(question, k=candidates), k=k)


def retrieve(question, k=4, method="hybrid"):
    if method == "vector":
        return search(question, k=k)
    if method == "keyword":
        return keyword_search(question, k=k)
    return hybrid_search(question, k=k)


# --- Part 2b. Answering with citations ------------------------------------------------

ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below, which come from a \
software project's documentation.

Rules:
- After every fact, cite the source it came from, like [1] or [2][3].
- If the sources don't contain the answer, reply exactly: I don't know based on the project documents.
- Don't use outside knowledge. Keep the answer short (a few sentences).

{sources}

Question: {question}
Answer:"""

NOT_FOUND = "I don't know based on the project documents."
CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(question, hits):
    sources = "\n\n".join(f"[{n}] ({hit.source} > {hit.heading})\n{hit.content}"
                          for n, hit in enumerate(hits, 1))
    return ANSWER_PROMPT.format(sources=sources, question=question)


def cited_numbers(answer, n_sources):
    """The source numbers the answer cites, in order, ignoring numbers that don't exist."""
    seen = []
    for number in map(int, CITATION_RE.findall(answer)):
        if 1 <= number <= n_sources and number not in seen:
            seen.append(number)
    return seen


def said_not_found(answer):
    return "don't know based on the project documents" in answer.lower().replace("’", "'")


@dataclass
class RagAnswer:
    answer: str
    model: config.ModelConfig
    sources: list = field(default_factory=list)   # the retrieved chunks, numbered from 1
    cited: list = field(default_factory=list)     # source numbers the answer cites
    not_found: bool = False                       # the model said the documents don't answer it
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: object = None


async def answer(question, k=4, method="hybrid", tier="cheap"):
    """Retrieve k chunks, then have a model answer from them, citing its sources."""
    hits = await asyncio.to_thread(retrieve, question, k, method)
    model = config.MODELS[tier]
    reply = await ask_model(model, build_prompt(question, hits))
    return RagAnswer(
        answer=reply.answer,
        model=model,
        sources=hits,
        cited=cited_numbers(reply.answer, len(hits)),
        not_found=said_not_found(reply.answer),
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        cost_usd=pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens),
    )