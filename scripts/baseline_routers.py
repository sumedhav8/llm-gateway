"""Compare simple routing strategies before training anything.

Usage (from the project folder):
    python -m scripts.baseline_routers               # newest labels file

Strategies:
- always strong / always cheap : the two extremes
- random                       : send a share of questions to cheap at random
- rules                        : hand-written rules about what looks "hard"
- oracle                       : perfect knowledge of which answers were good enough
                                 (impossible in practice; the best any router could do)
"""

import argparse
import random
import re

from scripts.routing_data import evaluate, format_row, load_items, newest_labels_file

# Words that suggest a question needs careful reasoning, calculation or long writing
HARD_WORDS = re.compile(
    r"\b(how many|how much|calculate|total|each|per|percent|average|explain|why|"
    r"compare|write|essay|story|poem|list|plan|step)\b",
    re.IGNORECASE,
)


def rules_send_to_cheap(question):
    """True if a question looks easy enough for the cheap model."""
    words = len(question.split())
    numbers = len(re.findall(r"\d+", question))
    if words > 40:          # long questions usually need more reasoning
        return False
    if numbers >= 2:        # several numbers usually means a multi-step calculation
        return False
    if HARD_WORDS.search(question):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_file", nargs="?", help="Labels file (default: newest)")
    args = parser.parse_args()

    labels_path = args.labels_file or newest_labels_file()
    items = load_items(labels_path)
    print(f"{len(items)} graded questions from {labels_path}\n")

    rng = random.Random(42)
    strategies = {
        "always strong": [False] * len(items),
        "always cheap": [True] * len(items),
        "random (34% to cheap)": [rng.random() < 0.34 for _ in items],
        "rules": [rules_send_to_cheap(item["question"]) for item in items],
        "oracle (perfect, impossible)": [item["good_enough"] for item in items],
    }
    for name, decisions in strategies.items():
        print(format_row(name, evaluate(items, decisions)))

    rules = strategies["rules"]
    print("\nHow the rules routed each category (share sent to cheap):")
    for category in sorted({item["category"] for item in items}):
        idx = [i for i, item in enumerate(items) if item["category"] == category]
        print(f"  {category:<17} {sum(rules[i] for i in idx)}/{len(idx)}")


if __name__ == "__main__":
    main()