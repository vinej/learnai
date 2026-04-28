"""
05 — Hybrid retrieval and reranking

Two ways to make retrieval significantly better than plain top-k by cosine.

  HYBRID SEARCH
    Run BOTH a lexical search (BM25) and a vector search. Fuse the scores
    (or ranks) into one ranked list. Catches exact-match queries that
    dense embeddings miss ("model number XJ-7").

  RERANKING
    Retrieve a wider set with cheap retrieval (top-50 or top-100), then use
    a heavier CROSS-ENCODER to score each (query, doc) pair more accurately.
    Drop everything below the top-k. Big precision gain at small extra cost.

Run: python 05_hybrid_and_reranking.py
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ───────────────────────────────────────────────────────────
# Documents — mix of conversational + technical
# ───────────────────────────────────────────────────────────
DOCS = [
    "The XJ-7 router supports up to 1 Gbps Wi-Fi 6 throughput.",
    "Restart your modem before troubleshooting Wi-Fi issues.",
    "Wi-Fi 6 (802.11ax) brings OFDMA and improved efficiency in dense networks.",
    "To reset your network, hold the button for ten seconds.",
    "The XJ-7 supports MU-MIMO and beamforming for multiple devices.",
    "Always update firmware to the latest stable release for security.",
    "The XJ-9 router has a 2.5GbE WAN port for higher throughput.",
    "Enterprise routers often include built-in VPN servers.",
    "Check our website for product compatibility lists before buying.",
    "Replace your network cables every five years.",
]


# ───────────────────────────────────────────────────────────
# 1. Vector search baseline
# ───────────────────────────────────────────────────────────
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embed_model.encode(DOCS, normalize_embeddings=True)


def vector_search(query: str, k: int = 5):
    q = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), DOCS[i]) for i in top]


# ───────────────────────────────────────────────────────────
# 2. BM25 (lexical)
# ───────────────────────────────────────────────────────────
def tokenize(s):
    return s.lower().split()


bm25 = BM25Okapi([tokenize(d) for d in DOCS])


def bm25_search(query: str, k: int = 5):
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), DOCS[i]) for i in top]


# ───────────────────────────────────────────────────────────
# 3. Hybrid: reciprocal-rank fusion (RRF)
# ───────────────────────────────────────────────────────────
# RRF score = Σ 1 / (k0 + rank_i). Doesn't need raw scores to be on the same
# scale. Empirically very strong as a default fusion method.
def rrf(rank_lists: list[list[int]], k0: int = 60) -> dict[int, float]:
    out = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k0 + rank)
    return out


def hybrid_search(query: str, k: int = 5):
    v_ranks = [i for i, _, _ in vector_search(query, k=20)]
    b_ranks = [i for i, _, _ in bm25_search(query, k=20)]
    fused = rrf([v_ranks, b_ranks])
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(i, fused[i], DOCS[i]) for i in top]


# ───────────────────────────────────────────────────────────
# 4. Cross-encoder reranker
# ───────────────────────────────────────────────────────────
# A cross-encoder takes (query, doc) AS A PAIR and produces a single
# relevance score. Much more accurate than embedding cosine because it sees
# both texts together — but slower (no bulk index search).
# Pattern: retrieve 50 candidates with FAISS, rerank to top-5 with this.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def vector_then_rerank(query: str, k_initial: int = 10, k_final: int = 5):
    initial = vector_search(query, k=k_initial)
    pairs = [(query, d) for _, _, d in initial]
    scores = reranker.predict(pairs)
    rescored = sorted(zip(initial, scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(s), text) for ((idx, _, text), s) in rescored[:k_final]]


# ───────────────────────────────────────────────────────────
# Compare on a query that mixes lexical + semantic
# ───────────────────────────────────────────────────────────
QUERY = "How fast is the XJ-7?"

print(f"query: {QUERY}\n")
print("--- vector only ---")
for i, s, d in vector_search(QUERY)[:5]:
    print(f"  {s:+.3f}  {d}")
print("\n--- BM25 only ---")
for i, s, d in bm25_search(QUERY)[:5]:
    print(f"  {s:+.3f}  {d}")
print("\n--- hybrid (RRF) ---")
for i, s, d in hybrid_search(QUERY)[:5]:
    print(f"  {s:+.4f}  {d}")
print("\n--- vector → rerank ---")
for i, s, d in vector_then_rerank(QUERY)[:5]:
    print(f"  {s:+.3f}  {d}")


# ───────────────────────────────────────────────────────────
# Picking a strategy
# ───────────────────────────────────────────────────────────
# • Conversational, free-form questions → dense alone is usually enough.
# • Mix of natural language and SKU/error codes → HYBRID. Big lift.
# • Quality matters more than 50 ms of latency → ADD A RERANKER.
# • Top-of-the-stack production system → all three: hybrid retrieve top-50,
#   rerank to top-10, send to LLM.


# ───────────────────────────────────────────────────────────
# Reranker model choices
# ───────────────────────────────────────────────────────────
# • cross-encoder/ms-marco-MiniLM-L-6-v2  — small (~90 MB), fast, strong English.
# • cross-encoder/ms-marco-MiniLM-L-12-v2 — larger, +1-2 nDCG points.
# • bge-reranker-v2-m3                    — multilingual, top of MTEB.
# • Cohere Rerank API                     — managed, very strong, paid.
