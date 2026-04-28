"""
Exercise 2 — Measure a reranker's precision lift

Build a small corpus where some docs are deliberately near-duplicates of the
TRUE relevant doc. Plain top-k cosine often pulls the duplicates instead of
the right doc. A cross-encoder reranker fixes that.

The script asserts:
  - precision@3 with reranker ≥ precision@3 without reranker.
  - the reranker lifts the precision by at least 0.05.

Run: python exercises/02_reranker_lift.py
"""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


# ───────────────────────────────────────────────────────────
# Corpus + queries with hand-labeled relevance
# ───────────────────────────────────────────────────────────
DOCS = [
    "FastAPI is a Python web framework for building APIs.",
    "FastAPI uses Pydantic for request and response models.",
    "Flask is a micro web framework for Python; predates FastAPI.",
    "Django is a full-stack Python web framework.",
    "JavaScript is the primary language of the web browser.",
    "Node.js runs JavaScript outside the browser using V8.",
    "Express is a minimal Node.js web framework.",
    "GraphQL is a query language for APIs.",
    "REST is an architectural style for web APIs.",
    "Pydantic provides data validation via Python type hints.",
    "Type hints in Python are checked statically by tools like mypy.",
    "FastAPI was created by Sebastián Ramírez.",
]

# (query, relevant doc indices)
QUERIES = [
    ("Who built FastAPI?", [11]),
    ("What does Pydantic do?", [9]),
    ("What is FastAPI?", [0, 1, 11]),
    ("Tell me about Express.", [6]),
]


def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if not retrieved:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


# ───────────────────────────────────────────────────────────
# Stage 1 — vector retrieval
# ───────────────────────────────────────────────────────────
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embed_model.encode(DOCS, normalize_embeddings=True)


def vector_top_k(query: str, k: int) -> list[int]:
    q = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    return list(np.argsort(scores)[::-1][:k])


# ───────────────────────────────────────────────────────────
# Stage 2 — cross-encoder rerank
# ───────────────────────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_top_k(query: str, candidates: list[int], k: int) -> list[int]:
    pairs = [(query, DOCS[i]) for i in candidates]
    scores = reranker.predict(pairs)
    rescored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in rescored[:k]]


# ───────────────────────────────────────────────────────────
# Compare
# ───────────────────────────────────────────────────────────
K = 3
INITIAL = 8           # how many to retrieve before reranking

p_vector = []
p_rerank = []

print(f"{'query':<40}  P@{K} vec  P@{K} rerank")
for q, relevant in QUERIES:
    candidates = vector_top_k(q, INITIAL)
    p_v = precision_at_k(candidates, relevant, K)

    after = rerank_top_k(q, candidates, K)
    p_r = precision_at_k(after, relevant, K)

    p_vector.append(p_v)
    p_rerank.append(p_r)
    print(f"{q:<40}  {p_v:.3f}     {p_r:.3f}")


vec_avg = sum(p_vector) / len(p_vector)
rer_avg = sum(p_rerank) / len(p_rerank)
print(f"\naverage P@{K}: vector={vec_avg:.3f}, rerank={rer_avg:.3f}")

assert rer_avg >= vec_avg, "reranker should not hurt precision"
assert rer_avg - vec_avg >= 0.05, (
    f"expected reranker to lift P@{K} by ≥ 0.05, got {rer_avg - vec_avg:.3f}"
)
print("reranker provides measurable lift ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Vary INITIAL (5, 10, 20, 50). Bigger pool → bigger potential lift,
#   higher latency.
# • Try `cross-encoder/ms-marco-MiniLM-L-12-v2` — should add another point
#   or two of precision.
# • Time the reranker. On CPU it's ~50ms/pair. Can you batch it?
# ─────────────────────────────────────────────────────────────────────────────
