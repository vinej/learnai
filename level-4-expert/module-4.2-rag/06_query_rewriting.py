"""
06 — Query rewriting

Real user queries are often:
  • short and ambiguous  ("rates?")
  • full of typos        ("hwo fast is the xj7")
  • phrased very differently from the docs

QUERY REWRITING uses an LLM to PRE-PROCESS the user's query into something
the retriever can match better:

  EXPANSION    — add synonyms, related terms.
  CLARIFICATION — turn "rates?" into "What are the API rate limits?"
  HYPOTHETICAL DOCUMENT (HyDE) — have the LLM generate a fake "ideal answer",
                                 then embed THAT instead of the question.
                                 Surprisingly effective.
  MULTI-QUERY  — generate 3-5 paraphrases, retrieve for each, merge results.

Run: python 06_query_rewriting.py    # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Tiny corpus and dense index
# ───────────────────────────────────────────────────────────
DOCS = [
    "Acme's Starter plan API allows 60 requests per minute per IP.",
    "Acme's Team plan API allows 600 requests per minute.",
    "Acme's Enterprise plan offers custom rate limits.",
    "Backups run nightly and are retained for 30 days.",
    "Two-factor authentication is free on all plans.",
    "SSO is restricted to Team and Enterprise plans.",
]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embedder.encode(DOCS, normalize_embeddings=True)


def search(query_emb: np.ndarray, k: int = 3):
    scores = emb @ query_emb
    top = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), DOCS[i]) for i in top]


def search_text(text: str, k: int = 3):
    q = embedder.encode([text], normalize_embeddings=True)[0]
    return search(q, k=k)


# ───────────────────────────────────────────────────────────
# 1. Plain retrieval — short / vague query
# ───────────────────────────────────────────────────────────
QUERY = "rates?"

print(f"=== plain retrieval for query: {QUERY!r} ===")
for s, d in search_text(QUERY):
    print(f"  {s:+.3f}  {d}")


# ───────────────────────────────────────────────────────────
# 2. Clarification — let the LLM rewrite
# ───────────────────────────────────────────────────────────
def rewrite(query: str) -> str:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=120,
        temperature=0,
        system=(
            "Rewrite the user's vague question into a complete, specific "
            "question suitable for a search engine. Output the rewritten "
            "question only — no preamble, no quotes."
        ),
        messages=[{"role": "user", "content": query}],
    )
    print_usage("rewrite", response.usage)
    return response.content[0].text.strip()


rewritten = rewrite(QUERY)
print(f"\nrewritten query: {rewritten!r}")
print(f"=== retrieval after rewrite ===")
for s, d in search_text(rewritten):
    print(f"  {s:+.3f}  {d}")


# ───────────────────────────────────────────────────────────
# 3. HyDE — Hypothetical Document Embeddings
# ───────────────────────────────────────────────────────────
# Generate a hypothetical IDEAL ANSWER for the question, then embed that
# instead of the question. Counterintuitive, but works because answers and
# documents speak the same "language", whereas questions use different
# phrasings.
def hyde(query: str) -> str:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=160,
        temperature=0,
        system=(
            "Pretend you are answering the user's question with a short factual "
            "passage. The passage may be hypothetical — that's fine. "
            "Output only the passage, 1-3 sentences, no preamble."
        ),
        messages=[{"role": "user", "content": query}],
    )
    print_usage("hyde", response.usage)
    return response.content[0].text.strip()


hyde_text = hyde(QUERY)
print(f"\nHyDE document: {hyde_text!r}")
print("=== retrieval using HyDE-embedded query ===")
for s, d in search_text(hyde_text):
    print(f"  {s:+.3f}  {d}")


# ───────────────────────────────────────────────────────────
# 4. Multi-query — paraphrases + result fusion
# ───────────────────────────────────────────────────────────
def paraphrase(query: str, n: int = 3) -> list[str]:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=200,
        temperature=0.7,
        system=(
            f"Generate {n} different paraphrases of the user's question, each "
            "phrased in a way that might match different document wordings. "
            "Output one paraphrase per line, no numbers, no bullet points."
        ),
        messages=[{"role": "user", "content": query}],
    )
    print_usage("paraphrase", response.usage)
    return [line.strip() for line in response.content[0].text.splitlines() if line.strip()][:n]


variants = paraphrase(QUERY)
print(f"\nparaphrases: {variants}")

# Run retrieval for each, RRF-fuse the results.
def rrf_merge(ranked_lists, k0=60):
    scores = {}
    for ranks in ranked_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0) + 1 / (k0 + r)
    return sorted(scores, key=scores.get, reverse=True)


ranked = []
for v in variants:
    q_emb = embedder.encode([v], normalize_embeddings=True)[0]
    scored = sorted(range(len(DOCS)), key=lambda i: -float(emb[i] @ q_emb))
    ranked.append(scored)

merged = rrf_merge(ranked)[:3]
print("=== retrieval after multi-query RRF fusion ===")
for i in merged:
    print(f"  {DOCS[i]}")


# ───────────────────────────────────────────────────────────
# When to invest in query rewriting
# ───────────────────────────────────────────────────────────
# Costs: each variant adds an embedding call AND another search. HyDE adds
# one LLM call. Multi-query: one LLM + N searches.
#
# Worth it when:
#   • Users tend to type short, ambiguous queries (search bars).
#   • Your corpus uses very different vocabulary than your queries.
#   • Plain top-k cosine misses obvious results.
#
# Skip it when:
#   • Queries are already specific and well-formed.
#   • Latency matters more than recall.
#   • You can fix the gap with a reranker (file 05).
