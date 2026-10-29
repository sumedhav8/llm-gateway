"""Shared helpers for the router: load labeled questions and score routing decisions.

A "routing decision" is True when a question goes to the CHEAP model and False
when it goes to the STRONG model.

How a decision is scored:
- Sent to cheap  -> it's a good answer if the cheap model was "good enough".
- Sent to strong -> it's a good answer if the strong model was correct
                    (known for math; assumed correct for everyday questions,
                    since we didn't grade the strong model's everyday answers).
- Cost is what the strong model actually cost on the questions sent to it.
  The cheap model is free.
"""

import json
from pathlib import Path

LABELS_DIR = Path("data/labels")
RESULTS_DIR = Path("data/results")


def newest_labels_file():
    files = sorted(LABELS_DIR.glob("labels_*.jsonl"))
    if not files:
        raise SystemExit("No labels yet. Run: python -m scripts.grade")
    return files[-1]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(labels_path):
    """Labeled questions, each with what it cost to send it to the strong model."""
    labels = read_jsonl(labels_path)

    # The labels file is named labels_<run>.jsonl; costs live in data/results/<run>.jsonl
    run_name = Path(labels_path).stem.removeprefix("labels_")
    results_path = RESULTS_DIR / f"{run_name}.jsonl"
    strong_cost = {}
    if results_path.exists():
        for row in read_jsonl(results_path):
            if row["tier"] == "strong" and row.get("cost_usd"):
                strong_cost[row["id"]] = float(row["cost_usd"])

    items = []
    for label in labels:
        if label["good_enough"] is None:
            continue  # couldn't be graded, so it can't be used for training or scoring
        items.append({
            "id": label["id"],
            "category": label["category"],
            "question": label["question"],
            "good_enough": bool(label["good_enough"]),
            "strong_ok": label["strong_correct"] if label["strong_correct"] is not None else True,
            "strong_cost": strong_cost.get(label["id"], 0.0),
        })
    return items


def evaluate(items, send_to_cheap):
    """Score a list of routing decisions (one True/False per item)."""
    total = len(items)
    good = 0
    cost = 0.0
    for item, cheap in zip(items, send_to_cheap):
        if cheap:
            good += item["good_enough"]
        else:
            good += item["strong_ok"]
            cost += item["strong_cost"]

    baseline_cost = sum(item["strong_cost"] for item in items)
    return {
        "share_cheap": sum(send_to_cheap) / total,
        "quality": good / total,
        "cost": cost,
        "savings": max(0.0, (baseline_cost - cost) / baseline_cost) if baseline_cost else 0.0,
    }


def format_row(name, result):
    return (f"{name:<28} {result['share_cheap']:>6.0%} to cheap   "
            f"quality {result['quality']:>6.1%}   cost ${result['cost']:.4f}   "
            f"saves {result['savings']:>4.0%}")