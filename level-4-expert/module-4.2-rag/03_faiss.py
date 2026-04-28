"""
03 — FAISS basics

FAISS (Facebook AI Similarity Search) is the de-facto local vector index.
You'll often see it inside higher-level frameworks too.

Three indexes you'll meet:

  IndexFlatIP / IndexFlatL2   — EXACT search by inner product / L2 distance.
                                Slowest, perfect recall. Fine up to ~10k docs.
  IndexHNSWFlat               — graph-based ANN. Very fast, ~98%+ recall,
                                no training needed. Modern default.
  IndexIVFFlat / IndexIVFPQ   — partition + quantize. Memory-efficient at
                                100M+ scale. Needs a training step.

Run: python 03_faiss.py
"""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a small corpus and embed it
# ───────────────────────────────────────────────────────────
docs = [
    "Python is a high-level, dynamically-typed programming language.",
    "Pandas is a Python library for tabular data manipulation.",
    "NumPy provides multi-dimensional arrays and fast linear algebra.",
    "Matplotlib produces static plots in a MATLAB-like API.",
    "PyTorch is an open-source deep-learning framework.",
    "FAISS implements approximate nearest neighbour search.",
    "Cats are independent companions.",
    "Dogs are loyal pets that need walks.",
    "Espresso is brewed by forcing hot water through fine grounds.",
    "Single-origin coffees highlight regional flavor characteristics.",
    "Soccer is the most-watched sport globally.",
    "Tennis is played on grass, clay, or hard courts.",
]
print(f"corpus: {len(docs)} docs")

embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embed.encode(docs, normalize_embeddings=True).astype("float32")
DIM = emb.shape[1]


# ───────────────────────────────────────────────────────────
# 1. Flat (exact) index — perfect recall, fine for small corpora
# ───────────────────────────────────────────────────────────
flat = faiss.IndexFlatIP(DIM)            # IP with normalized vectors == cosine
flat.add(emb)
print(f"\nIndexFlatIP: {flat.ntotal} vectors, dim {DIM}")


# ───────────────────────────────────────────────────────────
# 2. HNSW — fast, recall ~99%, no training
# ───────────────────────────────────────────────────────────
hnsw = faiss.IndexHNSWFlat(DIM, 32)      # M=32: graph connectivity
hnsw.hnsw.efConstruction = 64            # quality of graph build
hnsw.hnsw.efSearch = 64                  # quality vs speed at query time
hnsw.add(emb)
print(f"IndexHNSWFlat: {hnsw.ntotal} vectors")


# ───────────────────────────────────────────────────────────
# 3. IVF (inverted file) — partition into k cells; query a few
# ───────────────────────────────────────────────────────────
nlist = 4                                 # tiny corpus → tiny k
quantizer = faiss.IndexFlatIP(DIM)
ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(emb)
ivf.add(emb)
ivf.nprobe = 2                            # how many cells to search at query
print(f"IndexIVFFlat trained, nlist={nlist}, nprobe={ivf.nprobe}")


# ───────────────────────────────────────────────────────────
# Search them all on the same query
# ───────────────────────────────────────────────────────────
def search(index, query: str, k: int = 3):
    q = embed.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q, k)
    return [(docs[i], float(s)) for i, s in zip(idx[0], scores[0])]


query = "what library do I use for plotting in Python?"
print(f"\nquery: {query}\n")

for name, idx in [("flat", flat), ("hnsw", hnsw), ("ivf", ivf)]:
    print(f"  --- {name} ---")
    for d, s in search(idx, query):
        print(f"    {s:.3f}  {d}")
    print()


# ───────────────────────────────────────────────────────────
# Save and load
# ───────────────────────────────────────────────────────────
out = Path(__file__).parent / "scratch"
out.mkdir(exist_ok=True)
faiss.write_index(flat, str(out / "flat.index"))
print(f"saved: {out / 'flat.index'}")
loaded = faiss.read_index(str(out / "flat.index"))
print(f"loaded: {loaded.ntotal} vectors")


# ───────────────────────────────────────────────────────────
# Metadata filtering
# ───────────────────────────────────────────────────────────
# FAISS itself only stores vectors. To filter ("only docs from 2024", "only
# customer X"), keep a parallel mapping (id → metadata) and either:
#   1. Pre-filter ids, then ask FAISS for nearest IN that subset (use
#      IndexIDMap + an IDSelector — slow on big subsets).
#   2. Over-fetch (k=100), filter in Python, take top-k. Cheap.
#   3. Use a vector store that natively supports metadata filtering
#      (Qdrant, Milvus, Weaviate, pgvector with WHERE clauses).
#
# For most apps, option 2 is fine.


# ───────────────────────────────────────────────────────────
# Real-world choice
# ───────────────────────────────────────────────────────────
# • < 100k docs, single machine: IndexFlatIP. Easy to reason about.
# • Up to a few million: IndexHNSWFlat.
# • Tens of millions / memory-constrained: IndexIVFPQ (compressed vectors).
#
# Or: skip FAISS and use a managed store (pgvector, Pinecone, Qdrant Cloud).
# Same algorithms under the hood; you trade a small premium for
# replication, durability, metadata filtering, and not running infra.
