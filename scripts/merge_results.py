"""Combine the cheap-model answers from one run with the strong-model answers from another.

Useful when you change the cheap model: re-run only the cheap tier, then reuse the
strong answers you already paid for, instead of paying for them again.

Usage (from the project folder):
    python -m scripts.merge_results --cheap data/results/run_NEW.jsonl --strong data/results/run_OLD.jsonl

Writes data/results/run_<timestamp>_merged.jsonl, which grade and the router scripts can use.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("data/results")


def read_rows(path, tier):
    with Path(path).open(encoding="utf-8") as f:
        return [row for row in map(json.loads, filter(str.strip, f)) if row["tier"] == tier]


def merge(cheap_path, strong_path):
    cheap = read_rows(cheap_path, "cheap")
    strong_by_id = {row["id"]: row for row in read_rows(strong_path, "strong")}
    merged = []
    for row in cheap:
        merged.append(row)
        if row["id"] in strong_by_id:
            merged.append(strong_by_id[row["id"]])
    missing = [row["id"] for row in cheap if row["id"] not in strong_by_id]
    return merged, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cheap", required=True, help="Results file with the cheap-model answers to use")
    parser.add_argument("--strong", required=True, help="Results file with the strong-model answers to reuse")
    args = parser.parse_args()

    merged, missing = merge(args.cheap, args.strong)
    out_path = RESULTS_DIR / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_merged.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(merged)} rows to {out_path}")
    if missing:
        print(f"Warning: {len(missing)} questions have no strong answer to reuse: {missing[:5]} ...")


if __name__ == "__main__":
    main()