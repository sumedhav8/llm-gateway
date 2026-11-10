"""Index documents for RAG: split them into chunks, embed them and store them in pgvector.

Usage (from the project folder; Docker and Ollama must be running):
    python -m scripts.rag_index README.md DEVLOG.md
    python -m scripts.rag_index README.md DEVLOG.md --show 3     # also print the first 3 chunks

Re-running replaces the stored chunks of those files, so run it again after editing them.
"""

import argparse
from collections import Counter

from app import rag


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Markdown files to index")
    parser.add_argument("--show", type=int, default=0, help="Print the first N chunks")
    args = parser.parse_args()

    try:
        chunks = rag.index_files(args.files)
    except rag.RagError as exc:
        raise SystemExit(str(exc))

    sizes = [len(c.content) for c in chunks]
    print(f"Indexed {len(chunks)} chunks (chunk limit {rag.CHUNK_CHARS} characters)")
    for source, count in Counter(c.source for c in chunks).items():
        print(f"  {source:<20} {count} chunks")
    print(f"Chunk length: average {sum(sizes) / len(sizes):.0f}, shortest {min(sizes)}, longest {max(sizes)} characters")

    for chunk in chunks[:args.show]:
        print(f"\n--- {chunk.source} #{chunk.index} ({len(chunk.content)} characters) ---")
        print(chunk.content[:600])


if __name__ == "__main__":
    main()