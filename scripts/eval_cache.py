"""Choose the cache threshold from data: how many paraphrases does it catch, how many look-alikes slip through?

It fills a pretend cache with the original questions from data/cache_pairs.jsonl, then
looks up every paraphrase and look-alike, exactly like the gateway does (most similar
saved question, cosine similarity):

- paraphrase served from the cache       -> good (saved money, same answer)
- look-alike served from the cache       -> WRONG HIT (the user gets an answer to a different question)

Two rules are compared at every threshold:
  similarity only            hit if similarity >= threshold
  similarity + same numbers  ...and the two questions contain exactly the same numbers

Free: embeddings run in Ollama. No database needed.

Usage (from the project folder):
    python -m scripts.eval_cache
    python -m scripts.eval_cache --prefix "clustering: "     # try another nomic-embed-text task prefix

Writes evals/cache_threshold.csv and evals/cache_threshold.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from app import embeddings
from app.cache import same_numbers
from scripts.routing_data import read_jsonl

PAIRS_PATH = Path("data/cache_pairs.jsonl")
EVALS_DIR = Path("evals")
THRESHOLDS = [round(t, 2) for t in np.arange(0.70, 1.0001, 0.01)]


def unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def lookups(pairs, vectors):
    """For every paraphrase and look-alike: its most similar original, and how similar.

    vectors holds the embeddings of all originals, then all paraphrases, then all look-alikes.
    """
    n = len(pairs)
    originals, queries = unit(vectors[:n]), unit(vectors[n:])
    similarity = queries @ originals.T          # cosine similarity of every query to every original
    best = similarity.argmax(axis=1)

    rows = []
    for q, kind in enumerate(["paraphrase"] * n + ["lookalike"] * n):
        own = q % n
        match = int(best[q])
        text = pairs[own][kind]
        rows.append({
            "id": pairs[own]["id"], "category": pairs[own]["category"], "kind": kind, "text": text,
            "matched_own": match == own, "similarity": float(similarity[q, match]),
            "own_similarity": float(similarity[q, own]),
            "same_numbers": same_numbers(text, pairs[match]["original"]),
        })
    return rows


def sweep(rows):
    """Paraphrases caught and look-alikes wrongly served, at each threshold, for both rules."""
    paraphrases = [r for r in rows if r["kind"] == "paraphrase"]
    lookalikes = [r for r in rows if r["kind"] == "lookalike"]
    table = []
    for t in THRESHOLDS:
        for rule in ("similarity only", "similarity + same numbers"):
            def served(r):
                return r["similarity"] >= t and (rule == "similarity only" or r["same_numbers"])
            caught = sum(served(r) and r["matched_own"] for r in paraphrases)
            wrong = sum(served(r) for r in lookalikes) + sum(served(r) and not r["matched_own"] for r in paraphrases)
            table.append({"threshold": t, "rule": rule,
                          "paraphrases_caught": caught / len(paraphrases),
                          "lookalikes_served": sum(served(r) for r in lookalikes) / len(lookalikes),
                          "wrong_hits": wrong})
    return table


def best_threshold(table, rule, max_wrong_share):
    """The lowest threshold (most hits) whose look-alike share stays at or under max_wrong_share."""
    ok = [row for row in table if row["rule"] == rule and row["lookalikes_served"] <= max_wrong_share]
    return max(ok, key=lambda row: (row["paraphrases_caught"], -row["threshold"])) if ok else None


def plot(table, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib isn't installed, so no chart was drawn.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rule, style in (("similarity only", "-"), ("similarity + same numbers", "--")):
        rows = [r for r in table if r["rule"] == rule]
        ax.plot([r["threshold"] for r in rows], [100 * r["paraphrases_caught"] for r in rows],
                style, color="tab:blue", label=f"paraphrases caught ({rule})")
        ax.plot([r["threshold"] for r in rows], [100 * r["lookalikes_served"] for r in rows],
                style, color="tab:red", label=f"look-alikes wrongly served ({rule})")
    ax.set_xlabel("Cache threshold (cosine similarity)")
    ax.set_ylabel("% of questions")
    ax.set_title("Semantic cache: hits vs wrong hits")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", help="Embedding task prefix to try (default: the one in .env)")
    args = parser.parse_args()
    if args.prefix is not None:
        embeddings.EMBED_PREFIX = args.prefix

    pairs = read_jsonl(PAIRS_PATH)
    texts = [p["original"] for p in pairs] + [p["paraphrase"] for p in pairs] + [p["lookalike"] for p in pairs]
    print(f"{len(pairs)} originals, {len(pairs)} paraphrases, {len(pairs)} look-alikes "
          f"(prefix {embeddings.EMBED_PREFIX!r})\n")
    rows = lookups(pairs, embeddings.embed(texts))

    for kind in ("paraphrase", "lookalike"):
        for group, keep in (("math", lambda r: r["category"] == "math"),
                            ("everyday", lambda r: r["category"] != "math")):
            sims = [r["own_similarity"] for r in rows if r["kind"] == kind and keep(r)]
            print(f"{kind:<11} {group:<9} similarity to its original: "
                  f"median {np.median(sims):.3f}   min {min(sims):.3f}   max {max(sims):.3f}")

    table = sweep(rows)
    print(f"\n{'threshold':>9}  {'rule':<26} {'paraphrases caught':>19} {'look-alikes served':>19}")
    for row in table:
        if round(row["threshold"] * 100) % 2 == 0:
            print(f"{row['threshold']:>9.2f}  {row['rule']:<26} {row['paraphrases_caught']:>19.0%} "
                  f"{row['lookalikes_served']:>19.0%}")

    print()
    for rule in ("similarity only", "similarity + same numbers"):
        for limit in (0.0, 0.05):
            row = best_threshold(table, rule, limit)
            label = "no look-alikes" if limit == 0 else f"<= {limit:.0%} look-alikes"
            if row:
                print(f"{rule:<26} {label:<17} threshold {row['threshold']:.2f}: "
                      f"catches {row['paraphrases_caught']:.0%} of paraphrases")
            else:
                print(f"{rule:<26} {label:<17} no threshold works")

    print("\nMost dangerous look-alikes (most similar to a different question's saved answer):")
    for r in sorted((r for r in rows if r["kind"] == "lookalike"), key=lambda r: -r["similarity"])[:5]:
        numbers = "same numbers" if r["same_numbers"] else "numbers differ"
        print(f"  {r['similarity']:.3f}  ({numbers})  {r['text'][:90]}")

    EVALS_DIR.mkdir(exist_ok=True)
    with (EVALS_DIR / "cache_threshold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "rule", "paraphrases_caught", "lookalikes_served", "wrong_hits"])
        writer.writeheader()
        writer.writerows(table)
    plot(table, EVALS_DIR / "cache_threshold.png")


if __name__ == "__main__":
    main()