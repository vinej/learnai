"""
Exercise 2 — Cosine similarity over text

Goal: compare two short texts by representing them as bag-of-words vectors
and computing cosine similarity.

This is exactly how naive text retrieval works (TF-IDF style). Real RAG uses
*dense* embeddings from a neural network, but the cosine-similarity step is
the same.

Run: python exercises/02_cosine_similarity.py
"""

import re
from collections import Counter

import numpy as np


WORD_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def text_to_vector(text: str, vocab: list[str]) -> np.ndarray:
    """Build a count vector for `text` over the given vocabulary order."""
    counts = Counter(tokenize(text))
    return np.array([counts[w] for w in vocab], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity(text1: str, text2: str) -> float:
    vocab = sorted(set(tokenize(text1)) | set(tokenize(text2)))
    v1 = text_to_vector(text1, vocab)
    v2 = text_to_vector(text2, vocab)
    return cosine_similarity(v1, v2)


if __name__ == "__main__":
    pairs = [
        # near-duplicates
        ("The cat sat on the mat.",
         "The mat had a cat on it."),

        # same topic, different words
        ("The cat sat on the mat.",
         "A feline was resting on the rug."),

        # totally unrelated
        ("The cat sat on the mat.",
         "Quantum mechanics describes subatomic particles."),

        # identical
        ("The cat sat on the mat.",
         "The cat sat on the mat."),
    ]

    for t1, t2 in pairs:
        print(f"sim = {similarity(t1, t2):.3f}")
        print(f"  A: {t1}")
        print(f"  B: {t2}\n")


# ── Reflection ───────────────────────────────────────────────────────────────
# Notice: the second pair (cat/feline, mat/rug) gets a LOW score even though
# the meaning is the same. That's the limit of bag-of-words: it doesn't
# understand synonyms. Embedding models in Module 4.2 (RAG) fix this by
# mapping "cat" and "feline" to nearby vectors in latent space.
# ─────────────────────────────────────────────────────────────────────────────
