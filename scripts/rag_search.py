"""Try retrieval: show the chunks the RAG search finds for a question.

Usage (from the project folder; Docker and Ollama must be running):
    python -m scripts.rag_search "Why was the semantic cache turned off?"
    python -m scripts.rag_search "How much did the router save?" -k 5
"""

import argparse

from app import rag


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to show")
    args = parser.parse_args()

    try:
        hits = rag.search(args.question, k=args.k)
    except rag.RagError as exc:
        raise SystemExit(str(exc))
    if not hits:
        raise SystemExit("Nothing indexed yet. Run: python -m scripts.rag_index README.md DEVLOG.md")

    for rank, hit in enumerate(hits, 1):
        print(f"\n#{rank}  similarity {hit.similarity:.3f}   {hit.source} chunk {hit.index}")
        print(f"    section: {hit.heading}")
        print("    " + hit.content[:400].replace("\n", "\n    "))


if __name__ == "__main__":
    main()