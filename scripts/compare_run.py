"""Compare models side by side using a results file from run_questions.

Usage (from the project folder):
    python -m scripts.compare_run                 # uses the newest results file
    python -m scripts.compare_run data/results/run_20261027_150000.jsonl

The correctness check is rough: it only asks "does the answer contain the
reference answer?" Questions with no reference are shown as "?".
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("data/results")


def newest_results_file():
    files = sorted(RESULTS_DIR.glob("run_*.jsonl"))
    if not files:
        raise SystemExit("No results yet. Run: python -m scripts.run_questions --tiers cheap strong")
    return files[-1]


def normalize(text):
    return text.lower().replace("$", "").replace(",", "").strip()


def looks_correct(answer, reference):
    """True / False, or None when it can't be checked automatically."""
    if answer is None:
        return False
    if reference is None:
        return None
    return normalize(reference) in normalize(answer)


def mark(result):
    return {True: "OK", False: "WRONG", None: "?"}[result]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_file", nargs="?", help="Results file (default: newest)")
    args = parser.parse_args()

    path = Path(args.results_file) if args.results_file else newest_results_file()
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    by_question = defaultdict(dict)   # question id -> {tier: row}
    tiers = []
    for row in rows:
        by_question[row["id"]][row["tier"]] = row
        if row["tier"] not in tiers:
            tiers.append(row["tier"])

    print(f"Results file: {path}\n")

    # 1) Every question, side by side
    for qid, answers in by_question.items():
        first = next(iter(answers.values()))
        print(f"{qid}  [{first.get('category')}]  {first['question']}")
        print(f"    reference: {first.get('reference')}")
        for tier in tiers:
            row = answers.get(tier)
            if row is None:
                continue
            check = mark(looks_correct(row["answer"], row.get("reference")))
            short = (row["answer"] or row.get("error") or "").replace("\n", " ")[:90]
            print(f"    {tier:<6} {check:<5}  {short}")
        print()

    # 2) Totals per model
    print("Totals")
    print("------")
    for tier in tiers:
        tier_rows = [r for r in rows if r["tier"] == tier]
        checked = [looks_correct(r["answer"], r.get("reference")) for r in tier_rows]
        correct = sum(1 for c in checked if c is True)
        checkable = sum(1 for c in checked if c is not None)
        cost = sum(float(r["cost_usd"]) for r in tier_rows if r.get("cost_usd"))
        answered = [r for r in tier_rows if r["status_code"] == 200]
        avg_ms = sum(r["latency_ms"] for r in answered) / len(answered) if answered else 0
        print(f"{tier:<6} {tier_rows[0]['model']:<20} correct {correct}/{checkable} checkable   "
              f"avg {avg_ms:,.0f} ms   total cost ${cost:.6f}")

    # 3) The interesting ones: cheap was wrong, strong was right
    if "cheap" in tiers and "strong" in tiers:
        print("\nCheap model WRONG, strong model RIGHT (the questions the router should send to strong):")
        found = False
        for qid, answers in by_question.items():
            cheap, strong = answers.get("cheap"), answers.get("strong")
            if not cheap or not strong:
                continue
            if (looks_correct(cheap["answer"], cheap.get("reference")) is False
                    and looks_correct(strong["answer"], strong.get("reference")) is True):
                print(f"  - {qid}: {cheap['question']}")
                found = True
        if not found:
            print("  (none found automatically; check the '?' questions by reading the answers)")


if __name__ == "__main__":
    main()