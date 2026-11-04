"""Make a test set for the semantic cache: for each question, a paraphrase and a look-alike.

- paraphrase: same meaning, different words. The cache SHOULD reuse the answer.
- look-alike: almost the same words, but a detail changes so the right answer is different
  (another number, place, person...). The cache must NOT reuse the answer.

Sonnet writes both versions for questions sampled from data/questions.jsonl.
Read a few of the results yourself: an LLM-made test set should be spot-checked.

Usage (from the project folder):
    python -m scripts.make_cache_pairs --math 30 --everyday 30

Writes data/cache_pairs.jsonl (about $0.10)
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from app import config
from app.models import ModelError, ask_model
from scripts.routing_data import read_jsonl

QUESTIONS_PATH = Path("data/questions.jsonl")
OUT_PATH = Path("data/cache_pairs.jsonl")
SEED = 11

PROMPT = """You are helping test a cache for an AI assistant. Here is a question a user asked:

{question}

Write two new versions of it:
1. "paraphrase": the SAME question in different words. Someone reading both must agree they ask exactly the same thing and have exactly the same correct answer.
2. "lookalike": change ONE important detail (a number, a name, a place, a time, a condition) so the wording stays very similar but the correct answer is DIFFERENT.

Reply with only a JSON object, like this:
{{"paraphrase": "...", "lookalike": "..."}}"""


def parse_pair(text):
    """Read the JSON object from the model's reply, or None if it isn't valid."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        pair = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(pair.get("paraphrase"), str) or not isinstance(pair.get("lookalike"), str):
        return None
    return pair


async def make_one(model, item, semaphore):
    async with semaphore:
        try:
            reply = await ask_model(model, PROMPT.format(question=item["question"]))
        except ModelError as exc:
            print(f"  {item['id']}: failed ({exc.message[:60]})")
            return None
    pair = parse_pair(reply.answer)
    if pair is None:
        print(f"  {item['id']}: couldn't read the reply, skipped")
        return None
    return {"id": item["id"], "category": item["category"], "original": item["question"],
            "paraphrase": pair["paraphrase"].strip(), "lookalike": pair["lookalike"].strip()}


async def make_all(items):
    model = config.MODELS["strong"]
    if not model.is_configured:
        raise SystemExit("The strong model isn't set up. Add STRONG_API_KEY to .env.")
    semaphore = asyncio.Semaphore(5)
    results = await asyncio.gather(*(make_one(model, item, semaphore) for item in items))
    return [r for r in results if r]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--math", type=int, default=30)
    parser.add_argument("--everyday", type=int, default=30)
    args = parser.parse_args()

    questions = read_jsonl(QUESTIONS_PATH)
    rng = random.Random(SEED)
    math = rng.sample([q for q in questions if q["category"] == "math"], args.math)
    everyday = rng.sample([q for q in questions if q["category"] != "math"], args.everyday)

    print(f"Asking {config.MODELS['strong'].name} for {len(math) + len(everyday)} paraphrase/look-alike pairs ...")
    pairs = asyncio.run(make_all(math + everyday))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs to {OUT_PATH}\n")

    for pair in pairs[:2] + pairs[-2:]:
        print(f"original:   {pair['original'][:110]}")
        print(f"paraphrase: {pair['paraphrase'][:110]}")
        print(f"look-alike: {pair['lookalike'][:110]}\n")


if __name__ == "__main__":
    main()