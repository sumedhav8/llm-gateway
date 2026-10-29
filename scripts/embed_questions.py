"""Turn every labeled question into an embedding and save them for training.

Usage (from the project folder):
    ollama pull nomic-embed-text          # once
    python -m scripts.embed_questions     # newest labels file

Saves data/features/<labels file name>.npz
"""

import argparse
from pathlib import Path

import numpy as np

from app.embeddings import EMBED_MODEL, embed
from scripts.routing_data import load_items, newest_labels_file

FEATURES_DIR = Path("data/features")


def features_path(labels_path):
    return FEATURES_DIR / f"{Path(labels_path).stem}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_file", nargs="?", help="Labels file (default: newest)")
    args = parser.parse_args()

    labels_path = args.labels_file or newest_labels_file()
    items = load_items(labels_path)
    print(f"Embedding {len(items)} questions with {EMBED_MODEL} ...")

    vectors = np.array(embed([item["question"] for item in items]), dtype=np.float32)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = features_path(labels_path)
    np.savez(out_path, ids=np.array([item["id"] for item in items]), vectors=vectors)
    print(f"Saved {vectors.shape[0]} embeddings of {vectors.shape[1]} numbers each to {out_path}")


if __name__ == "__main__":
    main()