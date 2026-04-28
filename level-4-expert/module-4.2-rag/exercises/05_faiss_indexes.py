"""
Exercise 5 — Benchmark FAISS index types at scale

Build a synthetic corpus of 50,000 random embeddings and compare three
FAISS indices on:
  - build time
  - peak memory (rough)
  - query latency
  - recall@10 vs the exact (Flat) baseline

This shows you what's happening under the hood of every vector DB. pgvector,
Pinecone, Qdrant, Milvus — all use these same algorithm families.

Run: python exercises/05_faiss_indexes.py
"""

import time

import faiss
import numpy as np


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Synthetic corpus
# ───────────────────────────────────────────────────────────
N    = 50_000
DIM  = 384
NQ   = 100         # number of test queries
K    = 10

print(f"generating {N:,} synthetic embeddings of dim {DIM} ...")
data = rng.standard_normal((N, DIM)).astype("float32")
data /= np.linalg.norm(data, axis=1, keepdims=True)        # normalize for IP

queries = rng.standard_normal((NQ, DIM)).astype("float32")
queries /= np.linalg.norm(queries, axis=1, keepdims=True)


# ───────────────────────────────────────────────────────────
# Ground truth via exact Flat
# ───────────────────────────────────────────────────────────
flat = faiss.IndexFlatIP(DIM)
flat.add(data)
_, gt = flat.search(queries, K)


# ───────────────────────────────────────────────────────────
# Benchmark function
# ───────────────────────────────────────────────────────────
def bench(name: str, index, build_fn=None):
    t0 = time.perf_counter()
    if build_fn:
        build_fn(index)
    else:
        index.add(data)
    build_t = time.perf_counter() - t0

    # Warm up
    index.search(queries[:5], K)
    t0 = time.perf_counter()
    _, found = index.search(queries, K)
    query_t = (time.perf_counter() - t0) / NQ

    # recall@K vs ground truth
    overlaps = [len(set(gt[i]) & set(found[i])) for i in range(NQ)]
    recall = sum(overlaps) / (NQ * K)

    print(f"{name:<22}  build {build_t:>6.2f}s  "
          f"query {query_t * 1e3:>6.2f} ms/q  recall@{K}={recall:.3f}")
    return recall


print(f"\nrunning {NQ} queries × top-{K}")

# 1. Flat baseline (perfect recall)
r_flat = bench("IndexFlatIP", faiss.IndexFlatIP(DIM))


# 2. HNSW — graph-based ANN
def build_hnsw(idx):
    idx.hnsw.efConstruction = 64
    idx.add(data)


hnsw = faiss.IndexHNSWFlat(DIM, 32)
hnsw.metric_type = faiss.METRIC_INNER_PRODUCT
hnsw.hnsw.efSearch = 64
r_hnsw = bench("IndexHNSWFlat", hnsw, build_hnsw)


# 3. IVF — partition + flat
def build_ivf(idx):
    idx.train(data)
    idx.add(data)


ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(DIM), DIM, 256, faiss.METRIC_INNER_PRODUCT)
ivf.nprobe = 16
r_ivf = bench("IndexIVFFlat", ivf, build_ivf)


# 4. IVF + Product Quantization (memory-efficient)
def build_ivfpq(idx):
    idx.train(data)
    idx.add(data)


ivfpq = faiss.IndexIVFPQ(
    faiss.IndexFlatIP(DIM), DIM, 256,
    16,                 # M = number of subvectors
    8,                  # bits per subvector
    faiss.METRIC_INNER_PRODUCT,
)
ivfpq.nprobe = 16
r_ivfpq = bench("IndexIVFPQ", ivfpq, build_ivfpq)


# ───────────────────────────────────────────────────────────
# Sanity
# ───────────────────────────────────────────────────────────
assert r_flat == 1.0
assert r_hnsw  > 0.85, f"HNSW recall@{K} = {r_hnsw:.3f} too low"
assert r_ivf   > 0.80, f"IVF recall@{K}  = {r_ivf:.3f} too low"
print("\nbenchmarks pass ✅")


# ───────────────────────────────────────────────────────────
# What the numbers mean
# ───────────────────────────────────────────────────────────
# • Flat = exact, slowest. Use up to ~100k vectors.
# • HNSW = graph-based, fast, ~99% recall, ~3-4× memory of raw vectors.
#         The default for most modern vector stores.
# • IVF = partitioned, faster but with a tunable recall vs speed trade-off.
#         Good when you have memory but not enough for HNSW's graph.
# • IVFPQ = compressed. ~10-30× smaller than raw vectors at the cost of
#         5-10 points of recall. Used by Pinecone, Milvus for huge corpora.
#
# For "I want exact answers and have <100k docs", use Flat. For everything
# else, HNSW is the modern default.


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Sweep efSearch on HNSW (16, 32, 64, 128, 256) and plot recall vs latency.
# • Sweep nprobe on IVF (1, 4, 16, 64). Same plot.
# • Try N=500k. Look at peak memory (use psutil + Process().memory_info()).
# • Re-run with IndexHNSWPQ — graph + compression for memory-constrained big corpora.
# ─────────────────────────────────────────────────────────────────────────────
