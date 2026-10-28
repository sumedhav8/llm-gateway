"""Grade a results file: was each cheap-model answer good enough?

- Questions with a numeric reference (math): the last number in the answer
  must match the reference. Free and exact.
- Everything else: the strong model acts as a judge. Costs a little per question.

Writes data/labels/labels_<run name>.jsonl and prints the headline numbers.

Usage (from the project folder):
    python -m scripts.grade                                  # newest results file
    python -m scripts.grade data/results/run_20261028_150000.jsonl
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

from app import config, pricing
from app.models import ModelError, ask_model
from scripts.compare_run import newest_results_file

LABELS_DIR = Path("data/labels")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

JUDGE_PROMPT = """You are grading an AI assistant's answer.

Question:
{question}

Reference answer (written by a person; other wordings or details can also be correct):
{reference}

Assistant's answer:
{answer}

Is the assistant's answer correct and helpful enough that a user would be satisfied with it?
Reply with exactly one word on the first line: PASS or FAIL.
On the second line, give a one-sentence reason."""


def to_number(text):
    """Turn '1,234', '$30' or '30.00' into a number, or None if it isn't one."""
    if text is None:
        return None
    cleaned = str(text).replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def last_number(text):
    """The last number that appears in a piece of text, or None."""
    if not text:
        return None
    matches = NUMBER_RE.findall(text.replace(",", ""))
    return float(matches[-1]) if matches else None


def grading_method(row):
    """'number' when the reference is a number, otherwise 'judge'."""
    if row.get("grading"):
        return row["grading"]
    return "number" if to_number(row.get("reference")) is not None else "judge"


def number_matches(answer, reference):
    expected, got = to_number(reference), last_number(answer)
    return got is not None and expected is not None and abs(got - expected) < 1e-6


def parse_verdict(text):
    """Read the judge's reply: True for PASS, False for FAIL, None if unclear."""
    for line in (text or "").splitlines():
        word = line.strip().strip("*#`_ ").upper()
        if not word:
            continue
        if word.startswith("PASS"):
            return True
        if word.startswith("FAIL"):
            return False
        return None
    return None


async def judge(judge_model, row):
    """Ask the strong model whether the cheap answer is good enough."""
    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        reference=row.get("reference") or "(none provided)",
        answer=row["answer"],
    )
    reply = await ask_model(judge_model, prompt)
    lines = [line.strip() for line in reply.answer.splitlines() if line.strip()]
    reason = lines[1] if len(lines) > 1 else ""
    cost = pricing.request_cost(judge_model.name, reply.input_tokens, reply.output_tokens)
    return parse_verdict(reply.answer), reason, cost


async def grade(results_path):
    rows = [json.loads(line) for line in Path(results_path).open(encoding="utf-8") if line.strip()]
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["id"]][row["tier"]] = row

    judge_model = config.MODELS["strong"]
    labels, judge_cost = [], 0.0

    for i, (qid, answers) in enumerate(by_id.items(), 1):
        cheap, strong = answers.get("cheap"), answers.get("strong")
        if cheap is None:
            continue
        method = grading_method(cheap)
        reason = ""

        if cheap["status_code"] != 200:
            good_enough, reason = False, "The cheap model gave no answer"
        elif method == "number":
            good_enough = number_matches(cheap["answer"], cheap["reference"])
        else:
            if not judge_model.is_configured:
                raise SystemExit("The judge needs the strong model. Add STRONG_API_KEY to .env.")
            try:
                good_enough, reason, cost = await judge(judge_model, cheap)
                judge_cost += float(cost or 0)
            except ModelError as exc:
                good_enough, reason = None, f"Judge error: {exc.message}"

        strong_correct = None
        if strong and strong["status_code"] == 200 and method == "number":
            strong_correct = number_matches(strong["answer"], strong["reference"])

        labels.append({
            "id": qid,
            "category": cheap.get("category"),
            "question": cheap["question"],
            "reference": cheap.get("reference"),
            "cheap_answer": cheap["answer"],
            "strong_answer": strong["answer"] if strong else None,
            "method": method,
            "good_enough": good_enough,
            "judge_reason": reason,
            "strong_correct": strong_correct,
        })
        print(f"[{i}/{len(by_id)}] {qid:<12} {method:<6} good enough: {good_enough}")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"labels_{Path(results_path).stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")

    print_summary(labels, judge_cost, out_path)


def print_summary(labels, judge_cost, out_path):
    decided = [label for label in labels if label["good_enough"] is not None]
    passed = sum(1 for label in decided if label["good_enough"])
    print(f"\nSaved {len(labels)} labels to {out_path}")
    if decided:
        print(f"\nCheap model good enough: {passed}/{len(decided)} ({100 * passed / len(decided):.1f}%)")

    by_category = defaultdict(list)
    for label in decided:
        by_category[label["category"]].append(label["good_enough"])
    for category, values in sorted(by_category.items()):
        print(f"  {category:<17} {sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)")

    unclear = len(labels) - len(decided)
    if unclear:
        print(f"\n{unclear} answers could not be graded (judge error or unclear verdict).")
    strong_math = [label["strong_correct"] for label in labels if label["strong_correct"] is not None]
    if strong_math:
        print(f"Strong model math accuracy: {sum(strong_math)}/{len(strong_math)}")
    print(f"Judge cost: ${judge_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_file", nargs="?", help="Results file (default: newest)")
    args = parser.parse_args()
    asyncio.run(grade(args.results_file or newest_results_file()))


if __name__ == "__main__":
    main()