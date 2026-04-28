"""
08 — Sentence embeddings and semantic search

For retrieval (RAG, search, deduplication, clustering), you usually want a
SENTENCE-level vector — one fixed-size embedding per text.

Mean-pooling raw BERT outputs works but isn't great. The `sentence-transformers`
library trains models specifically to produce strong sentence embeddings
where cosine similarity reflects semantic similarity.

This file uses `all-MiniLM-L6-v2` (~80MB, 384-D embeddings, blazingly fast).

Run: python 08_sentence_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"embedding dim: {model.get_sentence_embedding_dimension()}")


# ───────────────────────────────────────────────────────────
# A small "knowledge base" to search
# ───────────────────────────────────────────────────────────
docs = [
    "Python is a popular programming language for data science.",
    "Cats are independent and often aloof companions.",
    "Soccer is the most-watched sport in the world.",
    "Paris is famous for the Eiffel Tower and its art museums.",
    "Machine learning models improve as they see more data.",
    "Dogs are loyal and require regular walks and attention.",
    "London's underground transit system is called the Tube.",
    "Neural networks learn parameters via gradient descent.",
    "Tokyo is the capital of Japan with a vibrant pop culture.",
    "Espresso is brewed by forcing hot water through fine coffee grounds.",
]

doc_emb = model.encode(docs, normalize_embeddings=True)
print(f"\nencoded {len(docs)} documents → matrix shape {doc_emb.shape}")


# ───────────────────────────────────────────────────────────
# Search via cosine similarity
# ───────────────────────────────────────────────────────────
def search(query: str, k: int = 3):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_emb @ q_emb                             # already normalized → dot = cos
    top = np.argsort(scores)[::-1][:k]
    return [(docs[i], float(scores[i])) for i in top]


queries = [
    "How do machine learning models work?",
    "Tell me about pet companions.",
    "What city is famous for art?",
    "How is coffee made?",
]

for q in queries:
    print(f"\nquery: {q}")
    for doc, score in search(q):
        print(f"  [{score:.3f}] {doc}")


# ───────────────────────────────────────────────────────────
# Why these embeddings work for retrieval
# ───────────────────────────────────────────────────────────
# Sentence-transformers are trained with CONTRASTIVE objectives: pairs of
# semantically-related sentences should have nearby embeddings, unrelated
# pairs should have distant embeddings. Common training data includes:
#   • Natural Language Inference (NLI) labelled pairs.
#   • Stack Exchange questions and their accepted answers.
#   • Web data with weak supervision.
# The result: cosine similarity ≈ semantic similarity.


# ───────────────────────────────────────────────────────────
# Production tips
# ───────────────────────────────────────────────────────────
# • For thousands of docs: NumPy or PyTorch matmul is fine.
# • For millions of docs: index with FAISS, Annoy, or pgvector. Sub-ms queries.
# • Consider encoding once and CACHING the embeddings on disk
#   (np.savez_compressed, or in a vector database).
# • Choose model by size/quality tradeoff:
#     - all-MiniLM-L6-v2     (~80MB, 384-D)  — fast default
#     - all-mpnet-base-v2    (~420MB, 768-D) — stronger, slower
#     - bge-large / nomic    — current SOTA on retrieval benchmarks
#     - multilingual-e5-base — cross-lingual support


# ───────────────────────────────────────────────────────────
# Beyond simple cosine search
# ───────────────────────────────────────────────────────────
# Hybrid retrieval = sparse (BM25 over keywords) + dense (embeddings).
# Reranking      = retrieve top-100 with a fast model, rerank top-10 with a
#                  cross-encoder (slow but precise). Module 4.2 RAG covers all this.
