"""
Exercise 5 — Build a semantic search over your own documents

Embed a small corpus with a sentence-transformer, then answer queries by
returning the top-k most similar docs.

The script ships with a small set of fake "knowledge base" docs and a few
queries with expected answers. You can swap in your own list.

Run: python exercises/05_semantic_search.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ───────────────────────────────────────────────────────────
# A tiny "knowledge base"
# ───────────────────────────────────────────────────────────
KNOWLEDGE = [
    {"id": "py-001", "text": "Python's GIL prevents multiple native threads from executing Python bytecodes at once."},
    {"id": "py-002", "text": "List comprehensions provide a concise way to create lists from iterables."},
    {"id": "py-003", "text": "Decorators wrap a function to extend its behavior without modifying it."},
    {"id": "py-004", "text": "Generators produce values lazily using the yield keyword."},
    {"id": "ml-001", "text": "Gradient descent updates parameters by stepping opposite to the gradient of the loss."},
    {"id": "ml-002", "text": "Cross-entropy loss is used for multi-class classification."},
    {"id": "ml-003", "text": "Dropout randomly zeros activations during training to reduce overfitting."},
    {"id": "ml-004", "text": "Batch normalization stabilizes training by normalizing layer inputs."},
    {"id": "db-001", "text": "PostgreSQL is a relational database with strong support for JSON and full-text search."},
    {"id": "db-002", "text": "An index speeds up queries at the cost of slower writes and more storage."},
    {"id": "db-003", "text": "Sharding distributes a database horizontally across multiple machines."},
    {"id": "web-001", "text": "REST APIs expose resources over HTTP using verbs like GET, POST, PUT, DELETE."},
    {"id": "web-002", "text": "GraphQL lets clients request exactly the fields they need in a single round trip."},
]
texts = [item["text"] for item in KNOWLEDGE]


# ───────────────────────────────────────────────────────────
# Build the index
# ───────────────────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
print(f"indexed {len(texts)} docs → matrix shape {emb.shape}")


# ───────────────────────────────────────────────────────────
# Search
# ───────────────────────────────────────────────────────────
def search(query: str, k: int = 3):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(scores)[::-1][:k]
    return [(KNOWLEDGE[i]["id"], KNOWLEDGE[i]["text"], float(scores[i])) for i in top]


# ───────────────────────────────────────────────────────────
# Test queries with expected top-1 doc IDs
# ───────────────────────────────────────────────────────────
TEST_CASES = [
    ("What stops Python from running threads in parallel?", "py-001"),
    ("How do I make my model less prone to overfit?",         "ml-003"),
    ("Tell me about lazy evaluation in Python.",              "py-004"),
    ("How do I make queries on my SQL database faster?",      "db-002"),
    ("API design where the client picks fields.",             "web-002"),
]

correct = 0
for query, expected_id in TEST_CASES:
    hits = search(query, k=3)
    top_id = hits[0][0]
    is_correct = top_id == expected_id
    correct += int(is_correct)
    print(f"\nquery:    {query}")
    print(f"expected: {expected_id}    got: {top_id}    {'✓' if is_correct else '✗'}")
    for did, text, score in hits:
        print(f"  [{score:.3f}] {did}: {text}")

print(f"\ntop-1 accuracy: {correct}/{len(TEST_CASES)}")
assert correct >= len(TEST_CASES) - 1, \
    f"at least {len(TEST_CASES) - 1}/{len(TEST_CASES)} top-1 should match"
print("semantic search works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace KNOWLEDGE with your own corpus (Markdown files, Slack export,
#   docs). Read them with pathlib and chunk long ones into ~200-token pieces.
# • Persist embeddings to disk (np.savez_compressed) so re-loading is instant.
# • Add a RERANKER: retrieve top-20, then rerank with a cross-encoder
#   (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Module 4.2 RAG.
# • Index with FAISS for sub-millisecond retrieval at million-doc scale.
# ─────────────────────────────────────────────────────────────────────────────
