"""Check the judge: grade a random sample yourself, then see how often you agree.

You see each question and the cheap model's answer WITHOUT the judge's verdict,
so your opinion isn't influenced. At the end you get the agreement rate.

Usage (from the project folder):
    python -m scripts.review_judge             # newest labels file, 20 answers
    python -m scripts.review_judge --n 30
"""

import argparse
import json
import random
from pathlib import Path

LABELS_DIR = Path("data/labels")
SEED = 7


def newest_labels_file():
    files = sorted(LABELS_DIR.glob("labels_*.jsonl"))
    if not files:
        raise SystemExit("No labels yet. Run: python -m scripts.grade")
    return files[-1]


def agreement(pairs):
    """pairs = [(your verdict, judge verdict), ...]. Returns the share that match."""
    if not pairs:
        return 0.0
    return sum(1 for mine, judge in pairs if mine == judge) / len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_file", nargs="?", help="Labels file (default: newest)")
    parser.add_argument("--n", type=int, default=20, help="How many answers to review")
    args = parser.parse_args()

    path = Path(args.labels_file) if args.labels_file else newest_labels_file()
    labels = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    judged = [label for label in labels if label["method"] == "judge" and label["good_enough"] is not None]
    if not judged:
        raise SystemExit("No judged answers in this file.")
    sample = random.Random(SEED).sample(judged, min(args.n, len(judged)))

    print(f"Reviewing {len(sample)} answers from {path}")
    print("For each one, type y (good enough), n (not good enough) or s (skip).\n")

    reviews = []
    for i, label in enumerate(sample, 1):
        print("=" * 70)
        print(f"[{i}/{len(sample)}] {label['id']}  ({label['category']})")
        print(f"\nQUESTION:\n{label['question']}")
        print(f"\nREFERENCE (a person's answer):\n{label['reference']}")
        print(f"\nCHEAP MODEL'S ANSWER:\n{label['cheap_answer']}\n")
        choice = ""
        while choice not in {"y", "n", "s"}:
            choice = input("Good enough? [y/n/s]: ").strip().lower()
        if choice != "s":
            reviews.append({"id": label["id"], "mine": choice == "y",
                            "judge": label["good_enough"], "judge_reason": label["judge_reason"]})

    out_path = LABELS_DIR / f"review_{path.stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for review in reviews:
            f.write(json.dumps(review, ensure_ascii=False) + "\n")

    rate = agreement([(r["mine"], r["judge"]) for r in reviews])
    print("=" * 70)
    print(f"\nYou and the judge agreed on {rate:.0%} of {len(reviews)} answers.")
    disagreements = [r for r in reviews if r["mine"] != r["judge"]]
    if disagreements:
        print("\nDisagreements (worth reading):")
        for r in disagreements:
            print(f"  - {r['id']}: you said {'PASS' if r['mine'] else 'FAIL'}, "
                  f"judge said {'PASS' if r['judge'] else 'FAIL'} ({r['judge_reason']})")
    print(f"\nSaved your reviews to {out_path}")


if __name__ == "__main__":
    main()