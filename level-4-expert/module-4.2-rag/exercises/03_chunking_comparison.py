"""
Exercise 3 — Compare 3 chunking strategies

For a fixed corpus of long documents and a labeled set of queries, measure
recall@3 under three chunking strategies:

  A. fixed-size 200 chars (no overlap)
  B. fixed-size 200 chars + 40 char overlap
  C. recursive paragraph → sentence (max 300 chars)

The script asserts strategy C ≥ strategy A on average recall@3.

Run: python exercises/03_chunking_comparison.py
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer


# ───────────────────────────────────────────────────────────
# A few "long" docs and labeled queries
# ───────────────────────────────────────────────────────────
DOCS = {
    "python": (
        "Python is a high-level dynamically-typed language. "
        "It was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability with the use of significant indentation. "
        "Common implementations include CPython, PyPy, and Jython.\n\n"

        "Python's package manager is pip and modern projects are usually configured via pyproject.toml. "
        "The Python Software Foundation maintains the language and runs PyCon. "
        "Type hints, introduced in PEP 484, help static analyzers like mypy."
    ),
    "ml": (
        "Machine learning is the study of algorithms that improve through experience. "
        "Supervised learning maps inputs to outputs given labeled data. "
        "Unsupervised learning finds structure in unlabeled data. "
        "Reinforcement learning concerns agents acting in environments to maximize reward.\n\n"

        "Common metrics for classification are accuracy, precision, recall, F1, ROC-AUC. "
        "For regression we use MAE, MSE, RMSE, R². "
        "Cross-validation gives more reliable estimates than a single split."
    ),
    "git": (
        "Git is a distributed version-control system. "
        "It was created by Linus Torvalds in 2005. "
        "Every clone holds the entire history of the project, enabling fast branching. "
        "GitHub, GitLab, and Bitbucket are popular hosted services built on Git.\n\n"

        "Common commands include git status, git add, git commit, git push, git pull. "
        "git revert creates a new commit that undoes a previous one. "
        "git reset --hard removes commits AND changes — use carefully."
    ),
}

# Each query → (substring of correct doc) used to grade retrieval.
QUERIES = [
    ("Who created Git?",                                "Linus Torvalds"),
    ("What does git revert do?",                        "creates a new commit"),
    ("What does Python's type-hint PEP introduce?",     "PEP 484"),
    ("Difference between supervised and reinforcement learning?",
                                                        "agents acting in environments"),
    ("What's RMSE used for?",                           "regression"),
    ("What's pip?",                                     "package manager"),
]


# ───────────────────────────────────────────────────────────
# Chunkers
# ───────────────────────────────────────────────────────────
def fixed(text, size=200):
    return [text[i:i + size] for i in range(0, len(text), size)]


def fixed_overlap(text, size=200, overlap=40):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def recursive(text, max_size=300):
    out = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in paras:
        if len(p) <= max_size:
            out.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > max_size:
                    if buf:
                        out.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                out.append(buf.strip())
    return out


# ───────────────────────────────────────────────────────────
# Embed chunks and evaluate
# ───────────────────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def evaluate_strategy(chunker, label):
    chunks = []
    for doc_id, text in DOCS.items():
        for i, c in enumerate(chunker(text)):
            chunks.append({"id": f"{doc_id}#{i}", "doc": doc_id, "text": c})

    emb = embedder.encode([c["text"] for c in chunks],
                          normalize_embeddings=True)

    recalls = []
    for query, target_substring in QUERIES:
        q_emb = embedder.encode([query], normalize_embeddings=True)[0]
        scores = emb @ q_emb
        top3 = np.argsort(scores)[::-1][:3]
        hit = any(target_substring in chunks[i]["text"] for i in top3)
        recalls.append(int(hit))

    avg = sum(recalls) / len(recalls)
    print(f"  {label:<22} chunks={len(chunks):>3}  avg_recall@3={avg:.3f}")
    return avg, recalls


print(f"comparing 3 chunking strategies on {len(QUERIES)} labeled queries")
a, _ = evaluate_strategy(lambda t: fixed(t, 200),                    "A. fixed-200")
b, _ = evaluate_strategy(lambda t: fixed_overlap(t, 200, 40),         "B. fixed-200 +40 overlap")
c, _ = evaluate_strategy(lambda t: recursive(t, 300),                 "C. recursive (par→sent)")

print(f"\nA: {a:.3f}    B: {b:.3f}    C: {c:.3f}")

assert c >= a - 0.05, "expected recursive ≥ fixed within noise; got recursive worse"
print("chunking comparison done ✅")


# ── Reflection ──────────────────────────────────────────────────────────────
# • Fixed chunking can split a relevant fact across two chunks → recall miss.
# • Overlap helps when the answer straddles a boundary (improvement is small
#   on this tiny set; bigger on real corpora).
# • Recursive is usually a small but consistent win because chunks tend to
#   correspond to natural ideas.
# • Many other variants exist: token-aware (use the LLM's tokenizer), semantic
#   (file 02), parent-document, sliding-window with reranking.
# ─────────────────────────────────────────────────────────────────────────────
