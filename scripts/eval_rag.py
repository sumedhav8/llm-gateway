"""Measure the RAG answers: are they correct, and do they stick to the retrieved sources?

For each saved question (data/rag_questions.jsonl):
1. The gateway's RAG answers it: retrieve k chunks (hybrid by default), then the cheap model
   answers from them with [n] citations.
2. Sonnet judges the answer against the passage the question was written from:
     CORRECT   does it answer the question correctly?
     GROUNDED  is every claim supported by the retrieved sources (nothing made up)?
Also counts how often the answer cites a source and how often it says "I don't know".

Cost: about $0.20 (Haiku answers + Sonnet judging). Results are saved, so read them freely.

Usage (from the project folder; Docker and Ollama must be running):
    python -m scripts.eval_rag
    python -m scripts.eval_rag --method vector --k 5

Writes data/rag_answers_<method>_k<k>.jsonl
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from app import config, pricing, rag
from app.models import ModelError, ask_model
from scripts.eval_retrieval import QUESTIONS_PATH

JUDGE_PROMPT = """You are grading an answer written by a documentation assistant.

Question:
{question}

Reference passage (the documentation the question was written from):
{reference}

Sources the assistant was given:
{sources}

Assistant's answer:
{answer}

Grade two things:
- CORRECT: does the answer correctly answer the question, consistent with the reference passage? \
An answer of "I don't know" is NOT correct.
- GROUNDED: is every claim in the answer supported by the sources the assistant was given? \
"I don't know" counts as grounded.

Reply in exactly this format:
CORRECT: YES or NO
GROUNDED: YES or NO
REASON: one short sentence"""


def parse_grades(text):
    """(correct, grounded, reason) from the judge's reply; None for a grade it didn't give."""
    def grade(name):
        match = re.search(rf"{name}\s*:\s*\**\s*(YES|NO)", text or "", re.IGNORECASE)
        return None if not match else match.group(1).upper() == "YES"
    reason = re.search(r"REASON\s*:\s*(.*)", text or "", re.IGNORECASE)
    return grade("CORRECT"), grade("GROUNDED"), reason.group(1).strip() if reason else ""


def summarize(rows):
    graded = [r for r in rows if r["correct"] is not None and r["grounded"] is not None]
    n = len(graded) or 1
    return {
        "answered": len(rows),
        "graded": len(graded),
        "correct": sum(r["correct"] for r in graded) / n,
        "grounded": sum(r["grounded"] for r in graded) / n,
        "not_found": sum(r["not_found"] for r in rows) / (len(rows) or 1),
        "with_citation": sum(bool(r["cited"]) for r in rows) / (len(rows) or 1),
        "retrieved_own_chunk": sum(r["retrieved_own_chunk"] for r in rows) / (len(rows) or 1),
    }


async def run(questions, k, method):
    judge = config.MODELS["strong"]
    if not judge.is_configured or not config.MODELS["cheap"].is_configured:
        raise SystemExit("Both models must be set up in .env (CHEAP_API_KEY and STRONG_API_KEY).")
    semaphore = asyncio.Semaphore(4)

    async def one(q):
        async with semaphore:
            start = time.perf_counter()
            try:
                result = await rag.answer(q["question"], k=k, method=method, tier="cheap")
            except (ModelError, rag.RagError) as exc:
                print(f"  failed: {q['question'][:60]} ({exc})")
                return None
            latency = time.perf_counter() - start
            sources = "\n\n".join(f"[{n}] {hit.content}" for n, hit in enumerate(result.sources, 1))
            prompt = JUDGE_PROMPT.format(question=q["question"], reference=q.get("reference", ""),
                                         sources=sources, answer=result.answer)
            try:
                verdict = await ask_model(judge, prompt)
            except ModelError as exc:
                print(f"  judge failed: {exc.message[:60]}")
                return None
        correct, grounded, reason = parse_grades(verdict.answer)
        return {
            "question": q["question"], "answer": result.answer,
            "expected": f"{q['source']} #{q['chunk_index']}",
            "retrieved": [f"{h.source} #{h.index}" for h in result.sources],
            "retrieved_own_chunk": any((h.source, h.index) == (q["source"], q["chunk_index"])
                                       for h in result.sources),
            "cited": result.cited, "not_found": result.not_found,
            "correct": correct, "grounded": grounded, "reason": reason,
            "answer_cost": float(result.cost_usd or 0),
            "judge_cost": float(pricing.request_cost(judge.name, verdict.input_tokens, verdict.output_tokens) or 0),
            "seconds": round(latency, 2),
        }

    return [r for r in await asyncio.gather(*(one(q) for q in questions)) if r]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["hybrid", "vector", "keyword"], default="hybrid")
    parser.add_argument("--k", type=int, default=4, help="Chunks given to the model")
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        raise SystemExit("No saved questions. Run: python -m scripts.eval_retrieval README.md DEVLOG.md")
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]

    # The judge needs the passage each question was written from: read it from the database
    try:
        stored = rag.stored_chunks()
    except rag.RagError as exc:
        raise SystemExit(str(exc))
    questions = [q for q in questions if (q["source"], q["chunk_index"]) in stored]
    for q in questions:
        q["reference"] = stored[(q["source"], q["chunk_index"])]

    print(f"Answering {len(questions)} questions ({args.method} retrieval, k={args.k}) and judging them ...")
    rows = asyncio.run(run(questions, args.k, args.method))
    s = summarize(rows)

    print(f"\nAnswered: {s['answered']}   graded: {s['graded']}")
    print(f"Correct:                   {s['correct']:.0%}")
    print(f"Grounded (nothing made up): {s['grounded']:.0%}")
    print(f"Said \"I don't know\":       {s['not_found']:.0%}")
    print(f"Cited at least one source: {s['with_citation']:.0%}")
    print(f"Own chunk was retrieved:   {s['retrieved_own_chunk']:.0%}")
    print(f"Cost: answers ${sum(r['answer_cost'] for r in rows):.4f}, judge ${sum(r['judge_cost'] for r in rows):.4f}; "
          f"average {sum(r['seconds'] for r in rows) / max(len(rows), 1):.1f} s per answer")

    wrong = [r for r in rows if r["correct"] is False]
    if wrong:
        print("\nAnswers judged wrong:")
        for r in wrong[:6]:
            own_chunk = "own chunk retrieved" if r["retrieved_own_chunk"] else "own chunk NOT retrieved"
            print(f"  [{own_chunk}] {r['question'][:70]}\n      judge: {r['reason'][:110]}")

    out_path = Path(f"data/rag_answers_{args.method}_k{args.k}.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved every answer to {out_path}")


if __name__ == "__main__":
    main()