"""
02 — Chunking strategies

Documents are usually too long to embed as a whole. CHUNKING splits them into
pieces small enough that each chunk has one focused meaning and fits in the
LLM's context.

Common strategies:

  FIXED SIZE              — every N characters/tokens. Simple, brittle on
                            mid-sentence cuts.
  FIXED + OVERLAP         — same, with K characters of overlap. Helps when a
                            split splits a sentence.
  RECURSIVE               — split by big delimiters first (paragraphs),
                            fall back to smaller (sentences) if too long.
                            The de facto default.
  SEMANTIC                — embed each sentence, group adjacent sentences
                            whose embeddings are similar. Slower, sometimes
                            cleaner.
  PARENT-DOCUMENT         — embed small chunks for retrieval, but RETURN the
                            larger parent paragraph/page to the LLM.
                            Preserves context.

Run: python 02_chunking.py
"""

import re
import textwrap


SAMPLE = """\
# How attention works

Attention scores are computed as the dot product of queries and keys, scaled
by the square root of the head dimension. The result is normalized via a
softmax to produce a probability distribution over positions.

The values are then multiplied by these attention weights, producing a
weighted sum that becomes the output of the attention layer. Each token
ends up as a learned mixture of every other token's value vector.

# Multi-head attention

Multi-head attention runs many attention computations in parallel, each on
a smaller subspace of the input. Concatenating the heads gives the layer
its capacity to attend to different relationships at once.

In practice, model architects pick the number of heads as a divisor of the
hidden dimension. 8 to 32 heads is common.

# Positional encodings

Transformers don't natively know token order. Positional encodings inject
that information either as fixed sinusoids, learned embeddings, or relative
schemes like RoPE and ALiBi.

RoPE in particular generalizes well to sequences longer than seen at
training time, and is used in modern LLMs like Llama and Qwen.
"""


# ───────────────────────────────────────────────────────────
# 1. Fixed-size chunking
# ───────────────────────────────────────────────────────────
def fixed_chunks(text: str, size: int) -> list[str]:
    return textwrap.wrap(text, width=size, replace_whitespace=False, drop_whitespace=False)


print("=== fixed-size 200 ===")
for c in fixed_chunks(SAMPLE, 200)[:3]:
    print(f"  [{len(c)}] {c[:60]!r}...")


# ───────────────────────────────────────────────────────────
# 2. Fixed-size with overlap
# ───────────────────────────────────────────────────────────
def fixed_overlap(text: str, size: int, overlap: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


print("\n=== fixed 200 with 40 overlap ===")
for c in fixed_overlap(SAMPLE, 200, 40)[:3]:
    print(f"  [{len(c)}] {c[:60]!r}...")


# ───────────────────────────────────────────────────────────
# 3. Recursive (paragraph → sentence → fixed)
# ───────────────────────────────────────────────────────────
def recursive_chunks(text: str, max_size: int = 400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paragraphs:
        if len(p) <= max_size:
            out.append(p)
        else:
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) + 1 > max_size:
                    if buf:
                        out.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                out.append(buf.strip())
    return out


print("\n=== recursive (paragraph → sentence) ===")
for c in recursive_chunks(SAMPLE, max_size=300):
    print(f"  [{len(c)}] {c[:60]!r}...")


# ───────────────────────────────────────────────────────────
# 4. Semantic chunking (sketch — uses embeddings)
# ───────────────────────────────────────────────────────────
def semantic_chunks(text: str, threshold: float = 0.6) -> list[str]:
    """Group adjacent sentences while their embeddings are similar."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return []

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(sentences, normalize_embeddings=True)

    chunks = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(emb[i] @ emb[i - 1])
        if sim >= threshold:
            chunks[-1] = chunks[-1] + " " + sentences[i]
        else:
            chunks.append(sentences[i])
    return chunks


print("\n=== semantic chunks (slower; embeds every sentence) ===")
for c in semantic_chunks(SAMPLE)[:3]:
    print(f"  [{len(c)}] {c[:60]!r}...")


# ───────────────────────────────────────────────────────────
# 5. Parent-document idea
# ───────────────────────────────────────────────────────────
# Build TWO indices:
#   • SMALL CHUNKS (200-300 chars) for embeddings → strong retrieval signal.
#   • PARENT (paragraph or page)   stored alongside, retrieved as context.
# At query time:
#   1. Embed the query, retrieve top-k SMALL chunks.
#   2. Look up each chunk's parent. Deduplicate parents.
#   3. Send the parents to the LLM (longer, context-rich).
# Best of both worlds: precise retrieval, context-rich generation.


# ───────────────────────────────────────────────────────────
# Picking sizes
# ───────────────────────────────────────────────────────────
# Goal: each chunk holds ONE idea. As a starting point:
#   200-400 chars (~50-100 tokens) for a tight passage.
#   500-1000 chars (~125-250 tokens) for paragraph-level retrieval.
#   1500+ chars only when parent-doc retrieval is in play.
# Always evaluate. A chunking change can swing retrieval P@k by 10+ points.
# Build a small eval set (file 07) before you ship.
