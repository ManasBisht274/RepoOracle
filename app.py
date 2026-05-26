"""
app.py — Streamlit UI for GitHub Repo Q&A (RAG with FAISS + GPT-4o).

Run:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Repo Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Validate API key ──────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ **OPENAI_API_KEY** not found. Copy `.env.example` → `.env` and add your key.")
    st.stop()

from src.loader import clone_repo, load_repo_files
from src.embedder import (
    build_vectorstore,
    chunk_documents,
    load_vectorstore,
    vectorstore_exists,
)
from src.chain import build_rag_chain

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 GitHub Repo Q&A")
    st.caption("Powered by LangChain · FAISS · GPT-4o")
    st.divider()

    st.subheader("📦 Load a Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/user/repo",
        value=st.session_state.repo_url,
    )

    col1, col2 = st.columns(2)
    ingest_btn = col1.button("🚀 Ingest", use_container_width=True, type="primary")
    load_btn = col2.button("📂 Load Existing", use_container_width=True)

    # ── Load existing index ───────────────────────────────────────────────────
    if load_btn:
        if not vectorstore_exists():
            st.error("No FAISS index found. Run ingestion first.")
        else:
            with st.spinner("Loading FAISS index..."):
                vs = load_vectorstore()
                st.session_state.chain, st.session_state.retriever = build_rag_chain(vs)
                st.session_state.indexed = True
            st.success("✅ Index loaded! Start asking questions.")

    # ── Ingest new repo ───────────────────────────────────────────────────────
    if ingest_btn:
        if not repo_url.strip():
            st.error("Please enter a GitHub repo URL.")
        elif not repo_url.startswith("https://github.com/"):
            st.error("URL must start with https://github.com/")
        else:
            st.session_state.repo_url = repo_url.strip()
            st.session_state.messages = []  # reset chat

            progress = st.progress(0, text="Cloning repository...")
            try:
                repo_path = clone_repo(repo_url.strip())
                progress.progress(25, text="Loading files...")

                documents = load_repo_files(repo_path)
                if not documents:
                    st.error("No supported source files found in this repo.")
                    st.stop()

                progress.progress(50, text=f"Chunking {len(documents)} files...")
                chunks = chunk_documents(documents)

                progress.progress(70, text="Building FAISS index (embedding)...")
                vs = build_vectorstore(chunks)

                progress.progress(90, text="Initializing RAG chain...")
                st.session_state.chain, st.session_state.retriever = build_rag_chain(vs)
                st.session_state.indexed = True

                progress.progress(100, text="Done!")
                st.success(f"✅ Indexed {len(documents)} files → {len(chunks)} chunks!")
                st.balloons()

            except Exception as exc:
                progress.empty()
                st.error(f"❌ Error during ingestion: {exc}")

    st.divider()

    # Status
    if st.session_state.indexed:
        st.success("🟢 Index ready")
        if st.session_state.repo_url:
            st.caption(f"Repo: `{st.session_state.repo_url.replace('https://github.com/', '')}`")
    else:
        st.warning("🟡 No index loaded")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with ❤️ using LangChain + FAISS + GPT-4o")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Chat with your Repository")

if not st.session_state.indexed:
    st.info(
        "👈 **Get started:** Enter a GitHub URL in the sidebar and click **Ingest**, "
        "or click **Load Existing** if you've already ingested a repo."
    )
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.code(src, language="text")

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask anything about the repository..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the answer
            for token in st.session_state.chain.stream(question):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Fetch sources separately
            source_docs = st.session_state.retriever.invoke(question)
            sources = sorted(
                {doc.metadata.get("source", "unknown") for doc in source_docs}
            )

            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for src in sources:
                        st.code(src, language="text")

        except Exception as exc:
            full_response = f"❌ Error: {exc}"
            response_placeholder.error(full_response)
            sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
