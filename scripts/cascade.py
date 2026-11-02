"""Evaluate a cascade: answer with the cheap model, check the answer, escalate if the check fails.

Unlike the router (which decides from the question alone), a cascade looks at the
cheap model's actual answer before deciding. Two checks are tried:

  A. Self-check: the cheap model rates how confident it is that the answer is
     correct, from 0 to 100. Accept the answer if the rating >= a threshold.
  B. Agreement (math only): ask the cheap model the same question a second time.
     If both final numbers match, accept; if not, escalate. Other questions use
     the self-check.

The checks never see the reference answer, just like in the live gateway.

Paid calls (one self-check per question, plus a second answer for math) are saved to
data/cascade/checks_<run>.jsonl, so re-running this script costs nothing.

Usage (from the project folder):
    python -m scripts.cascade              # newest labels file

Writes:
    evals/cascade_vs_router.csv
    evals/cascade_vs_router.png
"""

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app import config, pricing
from app.models import ModelError, ask_model
from scripts.grade import last_number
from scripts.routing_data import RESULTS_DIR, evaluate, load_items, newest_labels_file, read_jsonl
from scripts.train_router import best_row, load_training_data, make_model, threshold_curve

CHECKS_DIR = Path("data/cascade")
EVALS_DIR = Path("evals")
CONFIDENCE_THRESHOLDS = [0, 50, 60, 70, 80, 85, 90, 95, 100, 101]   # 101 = always escalate
PARALLEL_CALLS = 5

SELF_CHECK_PROMPT = """Here is a question and an answer written by an AI assistant.

Question:
{question}

Answer:
{answer}

How confident are you that this answer is correct and complete, so that a user would be satisfied with it?
Reply with only a whole number from 0 to 100."""


# --- Small helpers (tested in tests/test_cascade.py) ----------------------------

def parse_confidence(text):
    """The first whole number from 0 to 100 in the reply, or None."""
    for match in re.findall(r"\d+", text or ""):
        value = int(match)
        if 0 <= value <= 100:
            return value
    return None


def numbers_agree(answer_1, answer_2):
    """True when both answers end with the same number."""
    a, b = last_number(answer_1), last_number(answer_2)
    return a is not None and b is not None and abs(a - b) < 1e-6


def cascade_evaluate(items, accept, check_cost):
    """Score a cascade.

    accept[i]     True if the check accepted the cheap answer for item i.
    check_cost[i] what the check itself cost for item i.

    Every question pays for the cheap answer and the check; escalated questions
    also pay for the strong answer, and get the strong answer's quality.
    """
    total = len(items)
    good = 0
    cost = 0.0
    for item, ok, extra in zip(items, accept, check_cost):
        cost += item.get("cheap_cost", 0.0) + extra
        if ok:
            good += item["good_enough"]
        else:
            good += item["strong_ok"]
            cost += item["strong_cost"]

    baseline_cost = sum(item["strong_cost"] for item in items)
    return {
        "share_cheap": sum(accept) / total,
        "quality": good / total,
        "cost": cost,
        "savings": max(0.0, (baseline_cost - cost) / baseline_cost) if baseline_cost else 0.0,
    }


def to_float(cost):
    return float(cost) if cost is not None else 0.0


# --- Running the checks (paid; results are cached) ------------------------------

async def check_one(model, row, semaphore):
    """Self-check every answer; for math, also get a second answer to compare."""
    result = {"id": row["id"], "confidence": None, "self_check_cost": 0.0,
              "second_answer": None, "agree": None, "second_answer_cost": 0.0}
    async with semaphore:
        try:
            prompt = SELF_CHECK_PROMPT.format(question=row["question"], answer=row["answer"])
            reply = await ask_model(model, prompt)
            result["confidence"] = parse_confidence(reply.answer)
            result["self_check_cost"] = to_float(
                pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens))

            if row.get("category") == "math":
                reply = await ask_model(model, row["question"])
                result["second_answer"] = reply.answer
                result["agree"] = numbers_agree(row["answer"], reply.answer)
                result["second_answer_cost"] = to_float(
                    pricing.request_cost(model.name, reply.input_tokens, reply.output_tokens))
        except ModelError as exc:
            print(f"  {row['id']}: check failed ({exc.message[:80]}); it will be escalated")
    return result


async def run_checks(results_path, checks_path):
    model = config.MODELS["cheap"]
    if not model.is_configured:
        raise SystemExit("The cheap model isn't set up. Add CHEAP_API_KEY to .env.")

    rows = [r for r in read_jsonl(results_path) if r["tier"] == "cheap" and r.get("answer")]
    print(f"Checking {len(rows)} cheap answers with {model.name} ...")
    semaphore = asyncio.Semaphore(PARALLEL_CALLS)
    checks = await asyncio.gather(*(check_one(model, row, semaphore) for row in rows))

    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    with checks_path.open("w", encoding="utf-8") as f:
        for check in checks:
            f.write(json.dumps(check, ensure_ascii=False) + "\n")
    print(f"Saved checks to {checks_path}\n")


# --- Comparing strategies --------------------------------------------------------

def cascade_curves(items, checks):
    """Cost-vs-quality curve for each cascade design, one row per threshold."""
    self_check, agreement = [], []
    for t in CONFIDENCE_THRESHOLDS:
        accept_a, cost_a, accept_b, cost_b = [], [], [], []
        for item in items:
            c = checks.get(item["id"], {})
            confident = c.get("confidence") is not None and c["confidence"] >= t
            accept_a.append(confident)
            cost_a.append(c.get("self_check_cost", 0.0))

            if item["category"] == "math":
                accept_b.append(t <= 100 and bool(c.get("agree")))
                cost_b.append(c.get("second_answer_cost", 0.0))
            else:
                accept_b.append(confident)
                cost_b.append(c.get("self_check_cost", 0.0))
        self_check.append({"threshold": t, **cascade_evaluate(items, accept_a, cost_a)})
        agreement.append({"threshold": t, **cascade_evaluate(items, accept_b, cost_b)})
    return self_check, agreement


def router_curve(labels_path, items_by_id):
    """The trained router, scored with 5-fold cross-validation (same as train_router)."""
    router_items, X, y = load_training_data(labels_path)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(make_model(), X, y, cv=folds, method="predict_proba")[:, 1]
    # Re-order to match the cascade's items, so every design is scored on the same questions
    p_by_id = dict(zip([item["id"] for item in router_items], probabilities))
    ids = [i for i in items_by_id if i in p_by_id]
    return threshold_curve([items_by_id[i] for i in ids], [p_by_id[i] for i in ids]), ids


def plot(curves, points, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib isn't installed, so no chart was drawn.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in curves.items():
        ax.plot([100 * r["savings"] for r in curve], [100 * r["quality"] for r in curve],
                marker="o", markersize=3, label=name)
    for name, result in points.items():
        ax.scatter(100 * result["savings"], 100 * result["quality"], s=40, zorder=3, color="black")
        ax.annotate(name, (100 * result["savings"], 100 * result["quality"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Cost saved vs. always using the strong model (%)")
    ax.set_ylabel("Answers that are good (%)")
    ax.set_title("Cascade vs. router: cost saved vs. answer quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_file", nargs="?", help="Labels file (default: newest)")
    args = parser.parse_args()

    labels_path = Path(args.labels_file or newest_labels_file())
    run_name = labels_path.stem.removeprefix("labels_")
    results_path = RESULTS_DIR / f"{run_name}.jsonl"
    checks_path = CHECKS_DIR / f"checks_{run_name}.jsonl"

    if checks_path.exists():
        print(f"Reusing saved checks from {checks_path} (no new API calls)\n")
    else:
        asyncio.run(run_checks(results_path, checks_path))
    checks = {c["id"]: c for c in read_jsonl(checks_path)}

    items_by_id = {item["id"]: item for item in load_items(labels_path)}
    router, ids = router_curve(labels_path, items_by_id)
    items = [items_by_id[i] for i in ids]
    self_check, agreement = cascade_curves(items, checks)

    always_strong = evaluate(items, [False] * len(items))
    always_cheap = evaluate(items, [True] * len(items))
    oracle = evaluate(items, [item["good_enough"] for item in items])
    print(f"{len(items)} questions. Always strong: quality {always_strong['quality']:.1%}. "
          f"Always cheap: quality {always_cheap['quality']:.1%}, saves {always_cheap['savings']:.0%}.\n")

    # How well does each check catch the cheap model's mistakes?
    wrong = [item for item in items if not item["good_enough"]]
    caught = sum(1 for item in wrong if (checks.get(item["id"], {}).get("confidence") or 0) < 90)
    print(f"Self-check below 90 on {caught}/{len(wrong)} of the cheap model's wrong answers.")
    wrong_math = [item for item in wrong if item["category"] == "math"]
    disagreed = sum(1 for item in wrong_math if not checks.get(item["id"], {}).get("agree"))
    print(f"Two math answers disagreed on {disagreed}/{len(wrong_math)} of its wrong math answers.\n")

    designs = {"router (question only)": router,
               "cascade A: self-check": self_check,
               "cascade B: agreement + self-check": agreement}
    targets = [("no quality loss", 0.0), ("<=1% loss", 0.01), ("<=3% loss", 0.03), ("<=5% loss", 0.05)]

    print(f"{'design':<36}" + "".join(f"{name:>17}" for name, _ in targets))
    table = []
    for name, curve in designs.items():
        savings = []
        for _, drop in targets:
            row = best_row(curve, always_strong["quality"] - drop)
            savings.append(row["savings"] if row else 0.0)
        table.append([name] + savings)
        print(f"{name:<36}" + "".join(f"{'saves ' + format(s, '.0%'):>17}" for s in savings))

    print("\nCascade A at each threshold (accept the cheap answer if self-check >= threshold):")
    for row in self_check:
        print(f"  >= {row['threshold']:<3}  {row['share_cheap']:>5.0%} kept cheap   "
              f"quality {row['quality']:>6.1%}   cost ${row['cost']:.4f}   saves {row['savings']:>4.0%}")

    EVALS_DIR.mkdir(exist_ok=True)
    with (EVALS_DIR / "cascade_vs_router.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["design"] + [name for name, _ in targets])
        for row in table:
            writer.writerow([row[0]] + [round(s, 4) for s in row[1:]])
    plot(designs, {"always strong": always_strong, "always cheap": always_cheap, "oracle": oracle},
         EVALS_DIR / "cascade_vs_router.png")


if __name__ == "__main__":
    main()