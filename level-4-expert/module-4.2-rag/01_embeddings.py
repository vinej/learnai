"""
01 — Embeddings: from text to vectors

An EMBEDDING is a fixed-size vector that captures the SEMANTIC meaning of a
piece of text. Similar texts → nearby vectors.

Three flavors of embedding model:

  STATIC (word-level)     — word2vec, GloVe. Historical.
  SENTENCE-LEVEL          — sentence-transformers (MiniLM, MPNet, bge, e5).
                            Cheap, run locally, MTEB-strong.
  PROVIDER EMBEDDINGS     — voyage-3, cohere-embed-v3, openai/text-embedding-3.
                            Hosted; usually higher quality on long inputs.

Picking criteria:
  • Dimension                — 384 (MiniLM), 768 (MPNet), 1024 / 1536 (provider).
                               Higher = stronger, slower, more memory.
  • Speed                    — local MiniLM ≈ 5k docs/sec/CPU.
  • Quality on YOUR domain   — see MTEB leaderboard. Test on a sample.
  • Long-input handling      — provider models often allow 8k+ tokens.
  • Multilingual             — bge-m3, multilingual-e5 cross-lingual retrieval.

Run: python 01_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ───────────────────────────────────────────────────────────
# A small, fast, strong default
# ───────────────────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DIM = model.get_sentence_embedding_dimension()
print(f"model: all-MiniLM-L6-v2  dim={DIM}")


# ───────────────────────────────────────────────────────────
# Embed a few sentences
# ───────────────────────────────────────────────────────────
sentences = [
    "The cat is on the mat.",
    "A feline rests upon a rug.",                # similar meaning, different words
    "Stocks rallied on Wall Street today.",      # different topic
    "The S&P 500 reached an all-time high.",     # similar to the previous one
    "Quantum entanglement involves two particles.",
]

embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"\nshape: {embeddings.shape}    (5 sentences × {DIM} dim)")
print(f"each row already L2-normalized: norms = {np.linalg.norm(embeddings, axis=1).round(3)}")


# ───────────────────────────────────────────────────────────
# Cosine similarity (normalized → just dot product)
# ───────────────────────────────────────────────────────────
sim = embeddings @ embeddings.T
print(f"\ncosine similarity matrix (round to 2):")
for i, s in enumerate(sentences):
    print(f"  {i}: {s[:50]}")
print()
import pandas as pd
print(pd.DataFrame(sim.round(2)))


# ───────────────────────────────────────────────────────────
# A quick "search" demo
# ───────────────────────────────────────────────────────────
query = "How is the stock market doing?"
q_emb = model.encode([query], normalize_embeddings=True)[0]
scores = embeddings @ q_emb
order = np.argsort(scores)[::-1]
print(f"\nquery: {query}")
print("ranking:")
for rank, i in enumerate(order, start=1):
    print(f"  {rank}. score={scores[i]:.3f}  {sentences[i]}")


# ───────────────────────────────────────────────────────────
# Cosine vs dot product
# ───────────────────────────────────────────────────────────
# • If embeddings are L2-NORMALIZED, cosine = dot product. Fastest.
# • If they're NOT normalized (some provider models), use cosine explicitly:
#       cos(a, b) = a · b / (||a|| · ||b||)
# • Some models (older OpenAI ada-002) prefer raw dot product.
# Always read the model card.


# ───────────────────────────────────────────────────────────
# Asymmetric vs symmetric tasks
# ───────────────────────────────────────────────────────────
# RETRIEVAL is asymmetric: the query is short; documents are long. Some
# models distinguish "query" vs "passage" embeddings:
#
#   bge / e5 family expect prefixes:
#       query:    "query: How does X work?"
#       passage:  "passage: X works by ..."
#
# MiniLM and most general-purpose models don't need a prefix. Read the
# model card; using the wrong prefix can cost 10-30% retrieval accuracy.


# ───────────────────────────────────────────────────────────
# Storing embeddings
# ───────────────────────────────────────────────────────────
# • Tiny corpus (<10k docs) — keep them in a NumPy array. Search is a matmul.
# • Medium (10k-1M)         — use FAISS (file 03) on a single machine.
# • Big (1M+)               — managed vector store (Pinecone, Qdrant Cloud,
#                              pgvector with HNSW, Weaviate, Milvus).
#
# Common dimensions:
#   384  (MiniLM)              — 1.5 GB / 1M docs (fp32)
#   768  (MPNet)               — 3 GB
#   1024 (bge-large)           — 4 GB
#   1536 (text-embedding-3)    — 6 GB
# For big corpora, use float16 storage (cut by half) or product quantization.
