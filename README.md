# 🔍 GitHub Repo Q&A — RAG with LangChain + FAISS + GPT-4o

Ask natural language questions about **any GitHub repository**. The app clones the repo, indexes its source files using FAISS vector search, and answers your questions with GPT-4o — citing the exact files it pulled from.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange?logo=openai)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)

---

## ✨ Features

- 🔗 **Clone any public GitHub repo** with a single URL
- 📄 **Auto-loads 20+ file types** — Python, JS, TS, Go, Rust, Java, Markdown, YAML, and more
- ✂️ **Smart chunking** — different strategies for code vs. documentation
- 🔢 **OpenAI `text-embedding-3-small`** for fast, cheap embeddings
- 🗂️ **FAISS index** persisted locally — no re-embedding on restart
- 🤖 **GPT-4o** answers with streamed output and source file citations
- 🖥️ **Streamlit chat UI** with conversation history
- ⚡ **CLI ingestion** for scripting / CI pipelines

---

## 🏗️ Architecture

```
GitHub Repo URL
      │
      ▼
 ┌──────────┐     ┌───────────────┐     ┌────────────────┐
 │  GitPython│────▶│  File Loader  │────▶│ Text Splitter  │
 │  (clone)  │     │ (20+ ext)     │     │ (code / text)  │
 └──────────┘     └───────────────┘     └────────┬───────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │  OpenAI Embeddings      │
                                    │  text-embedding-3-small │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │  FAISS Vector Store     │
                                    │  (saved to disk)        │
                                    └────────────┬────────────┘
                                                 │
                         User Question           │ MMR Retrieval (k=6)
                               │                 │
                               ▼                 ▼
                        ┌──────────────────────────┐
                        │   LangChain RAG Chain     │
                        │   ChatPromptTemplate      │
                        │   GPT-4o (streaming)      │
                        └──────────────────────────┘
                                    │
                                    ▼
                            Answer + Source Files
```

---

## 🚀 Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/github-repo-qa.git
cd github-repo-qa
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and paste your OpenAI API key
```

### 5a. Ingest a repository (CLI)

```bash
python ingest.py https://github.com/tiangolo/fastapi
```

This clones the repo, loads files, embeds them, and saves the FAISS index.

### 5b. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter a GitHub URL in the sidebar, click **Ingest**, then start chatting!

---

## 📁 Project Structure

```
github-repo-qa/
├── src/
│   ├── __init__.py
│   ├── loader.py       # Clone repo + load source files
│   ├── embedder.py     # Chunk, embed, build/load FAISS index
│   └── chain.py        # LangChain RAG chain (GPT-4o)
├── app.py              # Streamlit chat UI
├── ingest.py           # CLI ingestion tool
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 💬 Example Questions

Once a repo is indexed, try asking:

- _"What does this project do? Give me a high-level overview."_
- _"How is authentication implemented?"_
- _"What are all the API endpoints and what do they do?"_
- _"Explain the database schema."_
- _"Where is error handling done and how?"_
- _"How do I run the tests for this project?"_
- _"What dependencies does this project use and why?"_

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `FAISS_INDEX_PATH` | `faiss_index/` | Where the FAISS index is saved |
| `MAX_FILE_SIZE_KB` | `500` | Skip files larger than this |
| Chunk size (code) | `800` tokens | In `embedder.py` |
| Chunk size (text) | `1200` tokens | In `embedder.py` |
| Retrieval `k` | `6` | Top-k docs retrieved per query |

---

## 🔧 Supported File Types

| Category | Extensions |
|---|---|
| Python | `.py` |
| JavaScript / TypeScript | `.js`, `.ts`, `.jsx`, `.tsx` |
| Systems | `.c`, `.cpp`, `.h`, `.hpp`, `.rs`, `.go` |
| JVM | `.java`, `.kt`, `.scala` |
| Web | `.html`, `.css`, `.scss` |
| Data / Config | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg` |
| Docs | `.md`, `.txt`, `.rst` |
| Scripts | `.sh`, `.bash`, `.sql`, `.graphql` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (CPU) |
| RAG Framework | LangChain 0.3 (LCEL) |
| Repo Cloning | GitPython |
| UI | Streamlit |

---

## 📝 License

MIT — feel free to fork and build on it!

---

> Made by [Manas Bisht](https://github.com/YOUR_USERNAME) • B.Tech CSE (AI & DS) • Graphic Era University
