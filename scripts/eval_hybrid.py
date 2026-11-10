"""Compare retrieval methods on the saved RAG questions: embeddings, keywords, or both (hybrid).

Uses the questions eval_retrieval.py saved in data/rag_questions.jsonl, and the chunks stored
in the database by rag_index.py (index the same files first). For each method, finds where
each question's own chunk ranks in the top 20, then reports recall@1/3/5 and MRR.

Free: embeddings run in Ollama, keyword search runs in Postgres.

Usage (from the project folder; Docker and Ollama must be running):
    python -m scripts.eval_hybrid

Writes evals/retrieval_hybrid.csv
"""

import csv
import json
from pathlib import Path

from app import rag
from scripts.eval_retrieval import EVALS_DIR, QUESTIONS_PATH, mrr, recall_at

DEPTH = 20                     # look this far down each ranking
METHODS = ["vector", "keyword", "hybrid"]


def rank_of(hits, source, index):
    """1-based rank of the chunk in the hits, or DEPTH + 1 if it isn't there."""
    for rank, hit in enumerate(hits, 1):
        if (hit.source, hit.index) == (source, index):
            return rank
    return DEPTH + 1


def main():
    if not QUESTIONS_PATH.exists():
        raise SystemExit("No saved questions. Run: python -m scripts.eval_retrieval README.md DEVLOG.md")
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]
    print(f"{len(questions)} questions, each method ranks the top {DEPTH} chunks\n")

    ranks = {method: [] for method in METHODS}
    try:
        for q in questions:
            for method in METHODS:
                hits = rag.retrieve(q["question"], k=DEPTH, method=method)
                ranks[method].append(rank_of(hits, q["source"], q["chunk_index"]))
    except rag.RagError as exc:
        raise SystemExit(str(exc))

    print(f"{'method':<10} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>6}")
    table = []
    for method in METHODS:
        r = ranks[method]
        row = {"method": method, "recall@1": recall_at(r, 1), "recall@3": recall_at(r, 3),
               "recall@5": recall_at(r, 5), "mrr": mrr(r)}
        table.append(row)
        print(f"{method:<10} {row['recall@1']:>9.0%} {row['recall@3']:>9.0%} {row['recall@5']:>9.0%} {row['mrr']:>6.2f}")

    fixed = [q for q, v, h in zip(questions, ranks["vector"], ranks["hybrid"]) if v > 5 >= h]
    broken = [q for q, v, h in zip(questions, ranks["vector"], ranks["hybrid"]) if h > 5 >= v]
    print(f"\nHybrid found {len(fixed)} questions in the top 5 that embeddings alone missed, "
          f"and lost {len(broken)} that embeddings found.")
    for q in fixed[:5]:
        print(f"  + {q['question'][:95]}")
    for q in broken[:5]:
        print(f"  - {q['question'][:95]}")

    EVALS_DIR.mkdir(exist_ok=True)
    with (EVALS_DIR / "retrieval_hybrid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "recall@1", "recall@3", "recall@5", "mrr"])
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSaved {EVALS_DIR / 'retrieval_hybrid.csv'}")


if __name__ == "__main__":
    main()