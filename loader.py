"""
loader.py — Clone a GitHub repo and load its source files as LangChain Documents.
"""

import os
import shutil
from pathlib import Path

from git import Repo
from langchain.schema import Document

# Extensions we care about — code + docs
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".r",
    ".md", ".txt", ".rst",
    ".yaml", ".yml", ".json", ".toml",
    ".cfg", ".ini", ".sh", ".bash",
    ".html", ".css", ".scss",
    ".sql", ".graphql",
    ".env.example", ".gitignore",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt",
    ".pytest_cache", ".mypy_cache",
    "coverage", ".tox", "eggs",
    ".eggs", "htmlcov",
}

MAX_FILE_SIZE_KB = 500  # Skip files larger than this


def clone_repo(repo_url: str, clone_dir: str = "repo_clone") -> str:
    """Clone a GitHub repo to a local directory. Removes existing clone first."""
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)

    print(f"📦 Cloning {repo_url} ...")
    Repo.clone_from(repo_url, clone_dir, depth=1)  # shallow clone = faster
    print(f"✅ Cloned into '{clone_dir}'")
    return clone_dir


def load_repo_files(repo_path: str) -> list[Document]:
    """
    Walk the cloned repo and load supported text files as LangChain Documents.
    Each Document carries metadata: source path, file type, file name.
    """
    documents: list[Document] = []
    repo_path = Path(repo_path)

    for file_path in repo_path.rglob("*"):
        # Skip unwanted directories
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        # Only files with supported extensions
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not file_path.is_file():
            continue
        # Skip very large files (e.g. vendored minified JS)
        if file_path.stat().st_size > MAX_FILE_SIZE_KB * 1024:
            print(f"⚠️  Skipping large file: {file_path.relative_to(repo_path)}")
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue

            relative_path = str(file_path.relative_to(repo_path))
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": relative_path,
                        "file_type": file_path.suffix.lower(),
                        "file_name": file_path.name,
                    },
                )
            )
        except Exception as exc:
            print(f"⚠️  Could not read {file_path}: {exc}")

    print(f"📄 Loaded {len(documents)} files from the repository.")
    return documents
