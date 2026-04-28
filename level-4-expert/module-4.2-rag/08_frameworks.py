"""
08 — Frameworks: when to use them, when to skip

LangChain, LlamaIndex, and Haystack each promise an end-to-end "build a RAG
in three lines" experience. Each has thousands of integrations.

The honest tradeoffs:

  YOU GET                                    YOU PAY
  - many loaders, splitters, retrievers      - heavy dependency tree
  - dozens of vector store integrations      - rapid API churn
  - prebuilt chains (chat-with-docs,         - debugging across abstraction layers
    agentic loops, etc.)                     - flexibility loss in the inner loop
  - eval frameworks, observability hooks      - opinionated abstractions
  - active community / docs

A 50-line hand-rolled pipeline beats a framework when:
  • You have a clear, narrow workflow (the kind covered in files 04, 05, 07).
  • You want to read every line in production.
  • You need precise control over prompts, latency, costs, telemetry.

A framework wins when:
  • You're rapidly prototyping across 5+ doc types and 3+ vector stores.
  • Your team will iterate on agentic flows where prebuilt patterns help.
  • You need a single library connecting disparate things you don't want to
    write yourself (Confluence loader, S3 watcher, parent-doc retriever, ...).

Run: python 08_frameworks.py
"""


# ───────────────────────────────────────────────────────────
# A vibe-check: same RAG pipeline, three idioms
# ───────────────────────────────────────────────────────────

HAND_ROLLED = """\
# 50-ish lines, no framework — you've already written this in file 04.
import anthropic, faiss, numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(docs, normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

def search(q, k=3):
    q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, idx = index.search(q_emb, k)
    return [docs[i] for i in idx[0]]

def answer(q):
    ctx = "\\n".join(f"[{i}] {d}" for i, d in enumerate(search(q)))
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        system="Answer using ONLY the documents. Cite [n].",
        messages=[{"role": "user", "content": f"{ctx}\\n\\nQuestion: {q}"}],
        max_tokens=400,
    ).content[0].text
"""


LLAMAINDEX = """\
# pip install llama-index llama-index-llms-anthropic
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Anthropic(model="claude-haiku-4-5-20251001")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = VectorStoreIndex.from_documents([Document(text=t) for t in docs])
query_engine = index.as_query_engine(similarity_top_k=3)
answer = query_engine.query("What's the refund policy?")
"""


LANGCHAIN = """\
# pip install langchain langchain-community langchain-anthropic
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents([Document(page_content=t) for t in docs], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# A real chain stitches these into a runnable pipeline (LCEL or RetrievalQA).
# This is where LangChain shines (and where you'll learn its abstractions).
"""


for label, code in [
    ("Hand-rolled", HAND_ROLLED),
    ("LlamaIndex", LLAMAINDEX),
    ("LangChain", LANGCHAIN),
]:
    print(f"\n--- {label} ---")
    print(code)


# ───────────────────────────────────────────────────────────
# A pragmatic recipe
# ───────────────────────────────────────────────────────────
# 1. PROTOTYPE on whichever framework matches your familiarity.
# 2. Once you understand the workflow, REWRITE the inner pipeline in plain
#    Python so you can debug, log, and tune freely.
# 3. Keep using the framework for parts where it really earns its keep —
#    document loaders, agent harnesses, integrations you don't want to maintain.


# ───────────────────────────────────────────────────────────
# Things you almost always want in production (regardless of framework)
# ───────────────────────────────────────────────────────────
# • Streaming     (file 4.1.03)
# • Citations     (file 4.2.04)
# • Caching       (file 4.1.06)
# • Eval harness  (file 4.2.07)
# • Telemetry: log every (query, retrieved chunks, prompt, answer, usage).
# • Cost dashboard: dollars per query, cache hit rate, P95 latency.
# • Index versioning: when you change embedding model or chunking, REBUILD
#   the index. Don't mix old and new vectors.


# ───────────────────────────────────────────────────────────
# Observability picks
# ───────────────────────────────────────────────────────────
# • Langfuse        — open-source, great UI for LLM traces.
# • LangSmith       — paid, deep LangChain integration.
# • Phoenix (Arize) — open-source, strong on RAG + evals.
# • Helicone        — proxy-based, drop-in for cost/latency.
# Even if you don't use a framework for orchestration, you should USE one
# of these (or build your own) to see what's happening on every call.
