"""Replay the labeled test questions through the RUNNING gateway with "tier": "auto".

The offline evaluation (scripts/cascade.py) estimated what routing would save. This
checks it for real: every question goes through the live gateway, which picks the
model, makes the calls and logs the request, exactly as for a real user.

How quality is scored:
- Math: the live answer is checked exactly (its last number vs the reference).
- Everyday questions: re-judging every live answer would cost extra, so the earlier
  labels are used: a Haiku answer counts as good if Haiku's labeled answer was good
  enough, and a Sonnet answer counts as good (as in the offline evaluation).

Usage (from the project folder; the gateway must be running):
    python -m scripts.replay --mode cascade
    python -m scripts.replay --mode router --url http://127.0.0.1:8001
    python -m scripts.replay --summarize data/replay/replay_cascade_20261103_150000.jsonl

Writes data/replay/replay_<mode>_<timestamp>.jsonl
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from scripts.grade import number_matches
from scripts.routing_data import load_items, newest_labels_file, read_jsonl

REPLAY_DIR = Path("data/replay")


async def ask_gateway(client, url, mode, item, semaphore):
    row = {"id": item["id"], "category": item["category"], "mode": mode}
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(f"{url}/ask", json={
                "question": item["question"], "tier": "auto", "mode": mode})
        except httpx.HTTPError as exc:
            row.update(status_code=None, error=f"Can't reach the gateway: {exc!r}")
            return row
        row["wall_ms"] = int((time.perf_counter() - start) * 1000)
    row["status_code"] = response.status_code
    if response.status_code == 200:
        row.update(response.json())
    else:
        row["error"] = response.text[:300]
    return row


async def replay(url, mode, items, parallel):
    semaphore = asyncio.Semaphore(parallel)
    async with httpx.AsyncClient(timeout=300) as client:
        tasks = [ask_gateway(client, url, mode, item, semaphore) for item in items]
        rows = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            rows.append(row)
            status = row.get("route") or f"error {row['status_code']}"
            print(f"[{i}/{len(items)}] {row['id']:<14} {status}")
    return rows


def is_good(row, label):
    """Was the answer the user got good? (see the note at the top of this file)"""
    if label.get("method") == "number":
        return number_matches(row["answer"], label["reference"])
    if row["tier"] == "cheap":
        return bool(label["good_enough"])
    return label["strong_correct"] if label.get("strong_correct") is not None else True


def summarize(rows, labels_by_id, strong_cost_by_id):
    ok = [row for row in rows if row.get("status_code") == 200]
    good = sum(is_good(row, labels_by_id[row["id"]]) for row in ok)
    cost = sum(row["cost_usd"] or 0.0 for row in ok)
    baseline = sum(strong_cost_by_id.get(row["id"], 0.0) for row in ok)
    latencies = [row["latency_ms"] for row in ok]
    return {
        "answered": len(ok),
        "errors": len(rows) - len(ok),
        "routes": Counter(row["route"] for row in ok),
        "share_cheap": sum(row["tier"] == "cheap" for row in ok) / len(ok) if ok else 0.0,
        "quality": good / len(ok) if ok else 0.0,
        "cost": cost,
        "always_strong_cost": baseline,
        "savings": max(0.0, 1 - cost / baseline) if baseline else 0.0,
        "avg_calls": sum(row["calls"] for row in ok) / len(ok) if ok else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def print_summary(summary, mode):
    print(f"\n=== Live replay, {mode} mode ===")
    print(f"Answered: {summary['answered']}   errors: {summary['errors']}")
    for route, count in summary["routes"].most_common():
        print(f"  {route:<38} {count}")
    print(f"Answered by Haiku:   {summary['share_cheap']:.0%}")
    print(f"Quality:             {summary['quality']:.1%}")
    print(f"Cost:                ${summary['cost']:.4f}  "
          f"(always Sonnet: ${summary['always_strong_cost']:.4f})  saves {summary['savings']:.0%}")
    print(f"Calls per question:  {summary['avg_calls']:.2f}")
    print(f"Latency:             avg {summary['avg_latency_ms']:,.0f} ms   p95 {summary['p95_latency_ms']:,.0f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["cascade", "router"], help="Routing mode to replay")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Where the gateway is running")
    parser.add_argument("--labels", help="Labels file (default: newest)")
    parser.add_argument("--limit", type=int, help="Only replay the first N questions")
    parser.add_argument("--parallel", type=int, default=4, help="Questions in flight at once")
    parser.add_argument("--summarize", help="Re-print the summary of an earlier replay file (no API calls)")
    args = parser.parse_args()

    labels_path = args.labels or newest_labels_file()
    labels_by_id = {label["id"]: label for label in read_jsonl(labels_path)
                    if label["good_enough"] is not None}
    items = [item for item in load_items(labels_path) if item["id"] in labels_by_id]
    strong_cost_by_id = {item["id"]: item["strong_cost"] for item in items}

    if args.summarize:
        rows = read_jsonl(args.summarize)
        print_summary(summarize(rows, labels_by_id, strong_cost_by_id), rows[0]["mode"])
        return
    if not args.mode:
        parser.error("choose --mode cascade or --mode router (or --summarize FILE)")

    items = items[:args.limit] if args.limit else items
    print(f"Replaying {len(items)} questions through {args.url} in {args.mode} mode ...")
    rows = asyncio.run(replay(args.url, args.mode, items, args.parallel))

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPLAY_DIR / f"replay_{args.mode}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(rows)} rows to {out_path}")

    if all(row.get("status_code") is None for row in rows):
        print(f"No answers: is the gateway running at {args.url}?")
        return
    print_summary(summarize(rows, labels_by_id, strong_cost_by_id), args.mode)


if __name__ == "__main__":
    main()