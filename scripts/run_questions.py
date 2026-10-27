"""Send a set of test questions to one or more models and save the results.

Usage (from the project folder):
    python -m scripts.run_questions --tiers cheap
    python -m scripts.run_questions --tiers cheap strong --limit 5

Results are saved to data/results/ as one JSON line per answer.
These runs are experiments, so they are NOT written to the requests table.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config, pricing
from app.models import ModelError, ask_model

DEFAULT_QUESTIONS = Path("data/questions_sample.jsonl")
RESULTS_DIR = Path("data/results")


def load_questions(path, limit=None):
    """Read questions from a .jsonl file (one JSON object per line)."""
    questions = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions[:limit] if limit else questions


async def run_one(model, item):
    """Ask one model one question and return a result row."""
    row = {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "reference": item.get("reference"),
        "tier": model.tier,
        "model": model.name,
    }
    start = time.perf_counter()
    try:
        reply = await ask_model(model, item["question"])
    except ModelError as exc:
        row.update(status_code=exc.status_code, answer=None, error=exc.message,
                   latency_ms=int((time.perf_counter() - start) * 1000),
                   input_tokens=None, output_tokens=None, cost_usd=None)
        return row

    cost = pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens)
    row.update(status_code=200, answer=reply.answer, error=None,
               latency_ms=int((time.perf_counter() - start) * 1000),
               input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
               cost_usd=str(cost) if cost is not None else None)
    return row


async def run(questions_path, tiers, limit):
    questions = load_questions(questions_path, limit)

    models_to_run = []
    for tier in tiers:
        model = config.MODELS[tier]
        if model.is_configured:
            models_to_run.append(model)
        else:
            print(f"Skipping the {tier} model: add {tier.upper()}_API_KEY to .env to include it.")
    if not models_to_run:
        print("No models to run.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl"
    rows = []

    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(questions, 1):
            for model in models_to_run:
                row = await run_one(model, item)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = "ok" if row["status_code"] == 200 else f"error {row['status_code']}"
                print(f"[{i}/{len(questions)}] {model.tier:<6} {status:<10} "
                      f"{row['latency_ms']:>7,} ms  {item['id']}")

    print(f"\nSaved {len(rows)} results to {out_path}\n")
    for model in models_to_run:
        ok = [r for r in rows if r["tier"] == model.tier and r["status_code"] == 200]
        total = sum(1 for r in rows if r["tier"] == model.tier)
        avg_ms = sum(r["latency_ms"] for r in ok) / len(ok) if ok else 0
        costs = [float(r["cost_usd"]) for r in ok if r["cost_usd"] is not None]
        print(f"{model.tier:<6} ({model.name}): {len(ok)}/{total} answered, "
              f"avg {avg_ms:,.0f} ms, cost ${sum(costs):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Path to a .jsonl questions file")
    parser.add_argument("--tiers", nargs="+", choices=["cheap", "strong"], default=["cheap"])
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N questions")
    args = parser.parse_args()
    asyncio.run(run(args.questions, args.tiers, args.limit))


if __name__ == "__main__":
    main()