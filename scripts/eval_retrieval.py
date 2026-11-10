"""Measure retrieval: for questions about the documents, is the right chunk found?

1. Haiku writes one question per sampled chunk ("what question does this passage answer?"),
   in its own words. Saved to data/rag_questions.jsonl and reused (about $0.01 the first time).
   Read some yourself: an LLM-made test set should be spot-checked.
2. Every chunk and question is embedded (free, Ollama), and each question's chunks are
   ranked by cosine similarity, the same math as the database search.
3. Scores, per embedding setup:
     recall@k  share of questions whose own chunk is in the top k
     MRR       mean reciprocal rank: 1 if the chunk is first, 1/2 if second, 1/3 if third...

Usage (from the project folder; Ollama must be running):
    python -m scripts.eval_retrieval README.md DEVLOG.md
    python -m scripts.eval_retrieval README.md DEVLOG.md --regenerate    # write new questions

Writes evals/retrieval.csv
"""

import argparse
import asyncio
import csv
import json
import random
from pathlib import Path

import numpy as np

from app import config, rag
from app.embeddings import embed
from app.models import ModelError, ask_model

QUESTIONS_PATH = Path("data/rag_questions.jsonl")
EVALS_DIR = Path("evals")
SEED = 5
MIN_CHARS = 200          # chunks shorter than this are too thin to ask about

QUESTION_PROMPT = """Here is a passage from a software project's documentation:

{passage}

Write ONE question that a reader could ask and that this passage answers. Use your own words: \
don't copy phrases from the passage. Reply with only the question."""

SETUPS = [
    ("search_document / search_query", rag.DOC_PREFIX, rag.QUERY_PREFIX),
    ("classification (like the router)", "classification: ", "classification: "),
    ("no prefix", "", ""),
]


def ranks(question_vectors, chunk_vectors, expected):
    """For each question, the rank (1 = best) of its own chunk among all chunks."""
    q = np.asarray(question_vectors, dtype=float)
    c = np.asarray(chunk_vectors, dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    similarity = q @ c.T
    order = np.argsort(-similarity, axis=1)
    return [int(np.where(order[i] == expected[i])[0][0]) + 1 for i in range(len(expected))]


def recall_at(rank_list, k):
    return sum(r <= k for r in rank_list) / len(rank_list)


def mrr(rank_list):
    return sum(1 / r for r in rank_list) / len(rank_list)


async def write_questions(chunks):
    model = config.MODELS["cheap"]
    if not model.is_configured:
        raise SystemExit("The cheap model isn't set up. Add CHEAP_API_KEY to .env.")
    semaphore = asyncio.Semaphore(5)

    async def one(chunk):
        async with semaphore:
            try:
                reply = await ask_model(model, QUESTION_PROMPT.format(passage=chunk.content))
            except ModelError as exc:
                print(f"  {chunk.source} #{chunk.index}: failed ({exc.message[:60]})")
                return None
        return {"source": chunk.source, "chunk_index": chunk.index, "heading": chunk.heading,
                "question": reply.answer.strip().strip('"')}

    return [q for q in await asyncio.gather(*(one(c) for c in chunks)) if q]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="The same markdown files you indexed")
    parser.add_argument("--n", type=int, default=40, help="How many chunks to write questions for")
    parser.add_argument("--regenerate", action="store_true", help="Write new questions even if saved ones exist")
    args = parser.parse_args()

    chunks = []
    for path in args.files:
        chunks += rag.chunk_markdown(Path(path).read_text(encoding="utf-8"), str(path).replace("\\", "/"))
    position = {(c.source, c.index): i for i, c in enumerate(chunks)}
    print(f"{len(chunks)} chunks from {len(args.files)} files")

    if QUESTIONS_PATH.exists() and not args.regenerate:
        questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]
        print(f"Reusing {len(questions)} saved questions from {QUESTIONS_PATH}")
    else:
        candidates = [c for c in chunks if len(c.content) >= MIN_CHARS]
        sample = random.Random(SEED).sample(candidates, min(args.n, len(candidates)))
        print(f"Asking {config.MODELS['cheap'].name} to write {len(sample)} questions ...")
        questions = asyncio.run(write_questions(sample))
        QUESTIONS_PATH.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions),
                                  encoding="utf-8")
        print(f"Saved them to {QUESTIONS_PATH}")

    questions = [q for q in questions if (q["source"], q["chunk_index"]) in position]
    if not questions:
        raise SystemExit("No saved question matches the current chunks. Run again with --regenerate.")
    expected = [position[(q["source"], q["chunk_index"])] for q in questions]

    print(f"\n{'embedding setup':<34} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>6}")
    table, best = [], None
    for name, doc_prefix, query_prefix in SETUPS:
        chunk_vectors = embed([c.content for c in chunks], prefix=doc_prefix)
        question_vectors = embed([q["question"] for q in questions], prefix=query_prefix)
        rank_list = ranks(question_vectors, chunk_vectors, expected)
        row = {"setup": name, "recall@1": recall_at(rank_list, 1), "recall@3": recall_at(rank_list, 3),
               "recall@5": recall_at(rank_list, 5), "mrr": mrr(rank_list)}
        table.append(row)
        print(f"{name:<34} {row['recall@1']:>9.0%} {row['recall@3']:>9.0%} {row['recall@5']:>9.0%} {row['mrr']:>6.2f}")
        if best is None:
            best = (rank_list, chunk_vectors, question_vectors)

    rank_list = best[0]
    print(f"\nQuestions whose chunk was NOT in the top 3 ({SETUPS[0][0]}):")
    misses = [(q, r) for q, r in zip(questions, rank_list) if r > 3]
    for q, r in misses[:8]:
        print(f"  rank {r:>3}  [{q['source']} > {q['heading'][:45]}]  {q['question'][:80]}")
    if not misses:
        print("  none")

    EVALS_DIR.mkdir(exist_ok=True)
    with (EVALS_DIR / "retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setup", "recall@1", "recall@3", "recall@5", "mrr"])
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSaved {EVALS_DIR / 'retrieval.csv'}")


if __name__ == "__main__":
    main()