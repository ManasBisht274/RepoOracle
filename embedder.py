"""
embedder.py — Chunk documents, embed with OpenAI, store/load FAISS index.
"""

import os

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

FAISS_INDEX_PATH = "faiss_index"

# ── Text splitter ────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.
    Code files use smaller chunks; markdown/text use slightly larger ones.
    """
    code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"}

    code_docs = [d for d in documents if d.metadata.get("file_type") in code_extensions]
    text_docs = [d for d in documents if d.metadata.get("file_type") not in code_extensions]

    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = (
        code_splitter.split_documents(code_docs)
        + text_splitter.split_documents(text_docs)
    )
    print(f"✂️  Split into {len(chunks)} chunks.")
    return chunks


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed chunks and persist the FAISS index to disk."""
    embeddings = _get_embeddings()
    print("🔢 Embedding chunks and building FAISS index (this may take a minute)...")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"💾 FAISS index saved to '{FAISS_INDEX_PATH}/'.")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously built FAISS index from disk."""
    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def vectorstore_exists() -> bool:
    """Check whether a FAISS index has already been built."""
    return os.path.isdir(FAISS_INDEX_PATH)
