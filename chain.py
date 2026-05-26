"""
chain.py — Build the RAG chain: retriever + GPT-4o + prompt.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert software engineer and codebase analyst.
You have been given relevant excerpts from a GitHub repository.

Guidelines:
- Answer questions about the code clearly and accurately.
- Always mention the file path (from metadata) when referencing specific code.
- If the code snippet is relevant, include it in a fenced code block with the correct language tag.
- If the answer isn't in the provided context, say: "I couldn't find that in the repository."
- Do not make up code or file paths.

Repository context:
──────────────────
{context}
──────────────────
"""

# ── Helper ────────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents for injection into the prompt."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        file_type = doc.metadata.get("file_type", "").lstrip(".")
        parts.append(f"📄 File: {source}\n```{file_type}\n{doc.page_content}\n```")
    return "\n\n".join(parts)


# ── Chain factory ─────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: FAISS):
    """
    Returns:
        chain   — A streaming LCEL chain (question → answer string)
        retriever — The underlying FAISS retriever (for source display)
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # Maximal Marginal Relevance — diverse results
        search_kwargs={"k": 6, "fetch_k": 12},
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Parallel branch so we can return both the answer AND source docs
    setup = RunnableParallel(
        context=retriever | _format_docs,
        question=RunnablePassthrough(),
    )

    chain = setup | prompt | llm | StrOutputParser()

    return chain, retriever
