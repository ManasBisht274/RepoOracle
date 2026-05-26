"""
ingest.py — CLI tool to clone a GitHub repo and build the FAISS index.

Usage:
    python ingest.py https://github.com/user/repo
    python ingest.py https://github.com/user/repo --clone-dir my_clone
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Validate API key early
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set. Copy .env.example → .env and add your key.")
    sys.exit(1)

from src.loader import clone_repo, load_repo_files
from src.embedder import chunk_documents, build_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repository into a local FAISS vector index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python ingest.py https://github.com/tiangolo/fastapi",
    )
    parser.add_argument(
        "repo_url",
        help="Full GitHub repository URL (e.g. https://github.com/user/repo)",
    )
    parser.add_argument(
        "--clone-dir",
        default="repo_clone",
        metavar="DIR",
        help="Local directory to clone the repo into (default: repo_clone)",
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting ingestion for: {args.repo_url}\n")

    # Step 1 — Clone
    repo_path = clone_repo(args.repo_url, args.clone_dir)

    # Step 2 — Load files
    documents = load_repo_files(repo_path)
    if not documents:
        print("❌ No supported files found in the repository. Exiting.")
        sys.exit(1)

    # Step 3 — Chunk
    chunks = chunk_documents(documents)

    # Step 4 — Embed + index
    build_vectorstore(chunks)

    print("\n✅ Ingestion complete!")
    print("👉 Run the app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
