"""Build a test question set from two public datasets.

Downloads (once) into data/raw/:
  - GSM8K test set: grade-school math word problems with exact numeric answers
    (https://github.com/openai/grade-school-math, MIT license)
  - Dolly 15k: everyday requests and answers written by people
    (https://huggingface.co/datasets/databricks/databricks-dolly-15k, CC BY-SA 3.0)

Writes data/questions.jsonl using a fixed random seed, so the sample is repeatable.

Usage (from the project folder):
    python -m scripts.build_dataset --math 100 --everyday 100
"""

import argparse
import json
import random
from pathlib import Path

import httpx

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/questions.jsonl")
SEED = 42

SOURCES = {
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "dolly_15k.jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
}

# Dolly categories that make sense as stand-alone questions (no extra text needed)
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing", "classification"}


def download(name, url):
    """Download a file into data/raw/ unless it is already there."""
    path = RAW_DIR / name
    if path.exists():
        return path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gsm8k_answer(solution):
    """GSM8K solutions end with '#### <number>'. Return just the number."""
    return solution.split("####")[-1].strip().replace(",", "")


def math_questions(rows, n, rng):
    picked = rng.sample(rows, n)
    return [
        {
            "id": f"gsm8k-{i:04d}",
            "category": "math",
            "source": "gsm8k",
            "grading": "number",
            "question": row["question"].strip(),
            "reference": gsm8k_answer(row["answer"]),
        }
        for i, row in enumerate(picked, 1)
    ]


def usable_everyday(rows):
    """Dolly rows that work as stand-alone questions of a sensible length."""
    return [
        row for row in rows
        if row["category"] in DOLLY_CATEGORIES
        and not row["context"].strip()
        and len(row["instruction"]) < 500
    ]


def everyday_questions(rows, n, rng):
    picked = rng.sample(usable_everyday(rows), n)
    return [
        {
            "id": f"dolly-{i:04d}",
            "category": row["category"],
            "source": "dolly",
            "grading": "judge",
            "question": row["instruction"].strip(),
            "reference": row["response"].strip(),
        }
        for i, row in enumerate(picked, 1)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--math", type=int, default=100, help="Number of GSM8K math questions")
    parser.add_argument("--everyday", type=int, default=100, help="Number of Dolly everyday questions")
    args = parser.parse_args()

    rng = random.Random(SEED)
    gsm8k = read_jsonl(download("gsm8k_test.jsonl", SOURCES["gsm8k_test.jsonl"]))
    dolly = read_jsonl(download("dolly_15k.jsonl", SOURCES["dolly_15k.jsonl"]))

    questions = math_questions(gsm8k, args.math, rng) + everyday_questions(dolly, args.everyday, rng)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_category = {}
    for item in questions:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
    print(f"Wrote {len(questions)} questions to {OUT_PATH}")
    for category, count in sorted(by_category.items()):
        print(f"  {category:<17} {count}")


if __name__ == "__main__":
    main()