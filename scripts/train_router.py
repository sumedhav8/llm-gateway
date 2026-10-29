"""Train the router: a classifier that predicts "is the cheap model good enough?"

It learns from question embeddings and the good-enough labels. With only a few
hundred questions, it is scored with 5-fold cross-validation: every question is
predicted by a model that never saw it during training.

Usage (from the project folder):
    python -m scripts.train_router            # newest labels file

Writes:
    evals/router_curve.csv        cost vs quality at each threshold
    evals/cost_vs_quality.png     the chart (if matplotlib is installed)
    data/models/router.joblib     the final model, trained on all questions
"""

import argparse
import csv
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from scripts.baseline_routers import rules_send_to_cheap
from scripts.embed_questions import features_path
from scripts.routing_data import evaluate, format_row, load_items, newest_labels_file

EVALS_DIR = Path("evals")
MODELS_DIR = Path("data/models")
THRESHOLDS = [round(t, 2) for t in np.arange(0.0, 1.0001, 0.05)]


def load_training_data(labels_path):
    items = load_items(labels_path)
    saved = np.load(features_path(labels_path))
    vector_by_id = dict(zip(saved["ids"].tolist(), saved["vectors"]))
    items = [item for item in items if item["id"] in vector_by_id]
    X = np.array([vector_by_id[item["id"]] for item in items])
    y = np.array([item["good_enough"] for item in items], dtype=int)
    return items, X, y


def make_model():
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")


def threshold_curve(items, probabilities):
    """Send a question to cheap when P(good enough) >= threshold."""
    rows = []
    for threshold in THRESHOLDS:
        decisions = [p >= threshold for p in probabilities]
        rows.append({"threshold": threshold, **evaluate(items, decisions)})
    return rows


def best_row(curve, min_quality):
    """The cheapest threshold that still keeps quality at or above min_quality."""
    ok = [row for row in curve if row["quality"] >= min_quality]
    return min(ok, key=lambda row: row["cost"]) if ok else None


def plot(curve, points, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib isn't installed, so no chart was drawn (pip install matplotlib).")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([100 * r["savings"] for r in curve], [100 * r["quality"] for r in curve],
            marker="o", markersize=3, label="Trained router (each dot = one threshold)")
    for name, result in points.items():
        ax.scatter(100 * result["savings"], 100 * result["quality"], s=40, zorder=3)
        ax.annotate(name, (100 * result["savings"], 100 * result["quality"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Cost saved vs. always using the strong model (%)")
    ax.set_ylabel("Answers that are good (%)")
    ax.set_title("Router: cost saved vs. answer quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_file", nargs="?", help="Labels file (default: newest)")
    args = parser.parse_args()

    labels_path = args.labels_file or newest_labels_file()
    items, X, y = load_training_data(labels_path)
    print(f"Training on {len(items)} questions ({y.sum()} where the cheap model was good enough)\n")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(make_model(), X, y, cv=folds, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probabilities)
    print(f"Cross-validated ROC AUC: {auc:.3f}  (0.5 = guessing, 1.0 = perfect)\n")

    curve = threshold_curve(items, probabilities)
    EVALS_DIR.mkdir(exist_ok=True)
    with (EVALS_DIR / "router_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "share_cheap", "quality", "cost", "savings"])
        writer.writeheader()
        writer.writerows(curve)

    rng = random.Random(42)
    points = {
        "always strong": evaluate(items, [False] * len(items)),
        "always cheap": evaluate(items, [True] * len(items)),
        "random": evaluate(items, [rng.random() < 0.34 for _ in items]),
        "rules": evaluate(items, [rules_send_to_cheap(item["question"]) for item in items]),
        "oracle": evaluate(items, [bool(label) for label in y]),
    }
    for name, result in points.items():
        print(format_row(name, result))

    print("\nTrained router at different thresholds:")
    for row in curve[::2]:
        print(format_row(f"threshold {row['threshold']:.2f}", row))

    strong_quality = points["always strong"]["quality"]
    print()
    for allowed_drop in (0.01, 0.03, 0.05):
        row = best_row(curve, strong_quality - allowed_drop)
        if row and row["savings"] >= 0.005:
            print(f"Giving up at most {allowed_drop:.0%} quality: threshold {row['threshold']:.2f} "
                  f"saves {row['savings']:.0%} of the cost ({row['share_cheap']:.0%} of questions to cheap).")
        else:
            print(f"Giving up at most {allowed_drop:.0%} quality: no threshold saves money yet.")

    plot(curve, points, EVALS_DIR / "cost_vs_quality.png")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model = make_model().fit(X, y)
    joblib.dump(final_model, MODELS_DIR / "router.joblib")
    print(f"Saved the final router model to {MODELS_DIR / 'router.joblib'}")


if __name__ == "__main__":
    main()
