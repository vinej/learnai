"""
04 — A complete RAG pipeline, end to end

The shape of every RAG system:

  INDEX TIME
    1. load documents
    2. chunk them
    3. embed each chunk
    4. store (chunk, embedding, metadata) in a vector index

  QUERY TIME
    1. embed the user's question
    2. retrieve top-k chunks
    3. build a prompt with the chunks as context
    4. call the LLM, return the answer (and the citations)

This file does the whole thing in ~80 lines, no framework.

Run: python 04_rag_pipeline.py    # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()


# ───────────────────────────────────────────────────────────
# 1. Documents
# ───────────────────────────────────────────────────────────
DOCUMENTS = [
    {"id": "policy-rt", "title": "Refunds",
     "text": "Acme refunds purchases pro-rata within 30 days. After 30 days, no refunds are issued."},
    {"id": "policy-2fa", "title": "Two-factor",
     "text": "All Acme customers may enable TOTP or WebAuthn-based two-factor authentication free of charge."},
    {"id": "policy-sso", "title": "SSO",
     "text": "SAML single sign-on is available on Team and Enterprise plans only. Free and Starter plans get standard auth."},
    {"id": "policy-rate", "title": "API rate limits",
     "text": "Starter: 60 requests per minute per IP. Team: 600 RPM. Enterprise: custom limits negotiated."},
    {"id": "policy-bk", "title": "Backups",
     "text": "Acme runs nightly incremental backups, retained for 30 days. Point-in-time recovery is Enterprise-only."},
    {"id": "policy-rg", "title": "Regions",
     "text": "Available regions: us-east, us-west, eu-west, ap-southeast. Data residency guarantees on Enterprise."},
    {"id": "policy-tr", "title": "Trial",
     "text": "Acme offers a 14-day free trial with full Team-plan features. No credit card required."},
]


# ───────────────────────────────────────────────────────────
# 2. Index time: embed + index
# ───────────────────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [d["text"] for d in DOCUMENTS]
emb = embedder.encode(texts, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
print(f"indexed {index.ntotal} docs (dim={emb.shape[1]})")


# ───────────────────────────────────────────────────────────
# 3. Query time: retrieve
# ───────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, k)
    return [{"score": float(s), **DOCUMENTS[i]} for i, s in zip(idx[0], scores[0])]


# ───────────────────────────────────────────────────────────
# 4. Build prompt + call LLM
# ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You answer questions about Acme using ONLY the given DOCUMENTS.

Rules:
- If the documents do not contain the answer, say "I don't know based on the provided documents."
- Cite the doc IDs in square brackets after each claim, like [policy-rt].
- Be concise. No preamble.
"""


def build_user_message(query: str, hits: list[dict]) -> str:
    docs_xml = "\n".join(
        f'<doc id="{h["id"]}" title="{h["title"]}">{h["text"]}</doc>'
        for h in hits
    )
    return (
        f"<documents>\n{docs_xml}\n</documents>\n\n"
        f"Question: {query}"
    )


client = anthropic.Anthropic()


def answer(query: str, k: int = 3):
    hits = retrieve(query, k=k)
    user_msg = build_user_message(query, hits)
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    print_usage("rag", response.usage)
    return response.content[0].text, hits


# ───────────────────────────────────────────────────────────
# 5. Try it
# ───────────────────────────────────────────────────────────
queries = [
    "What's the refund policy?",
    "Which plans support SAML SSO?",
    "How long are backups retained?",
    "Do you support cryptocurrency payments?",   # not in docs → 'I don't know'
]

for q in queries:
    print(f"\nQ: {q}")
    answer_text, hits = answer(q)
    print("retrieved:")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['id']}: {h['title']}")
    print("answer:", answer_text)


# ───────────────────────────────────────────────────────────
# Production additions
# ───────────────────────────────────────────────────────────
# This minimal pipeline is missing things you'd add for production:
#   • CHUNKING (file 02)        — for documents bigger than a paragraph.
#   • HYBRID + RERANK (file 05) — boosts top-k retrieval quality.
#   • QUERY REWRITE (file 06)   — clarify or expand vague user queries.
#   • EVALUATION (file 07)      — keep retrieval and answers honest.
#   • Streaming                  — start showing tokens immediately.
#   • Citations rendered as links — your UI converts [policy-rt] → URL.
#   • Prompt caching (4.1.06)    — system + retrieved docs reused = ~90% savings.
#   • Logging                    — every (query, retrieved, answer) for review.
