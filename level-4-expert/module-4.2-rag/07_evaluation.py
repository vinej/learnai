"""
07 — Evaluating RAG

You can't improve what you don't measure. RAG eval has TWO axes:

  RETRIEVAL QUALITY
    Did we surface the right documents?
    metrics: recall@k, precision@k, MRR, nDCG.

  ANSWER QUALITY
    Given the retrieved docs, did the model write a good answer?
    metrics:
    - faithfulness        (no hallucinations beyond the docs)
    - answer relevance    (does the answer match the question)
    - context relevance   (were the retrieved docs actually useful)

Two ways to grade ANSWER QUALITY:
  HUMAN EVAL (best, slow, expensive).
  LLM-AS-JUDGE (uses a frontier model with a clear rubric to score).

Frameworks: ragas, TruLens, DeepEval. Underneath each is "LLM-as-judge with
some pre-baked rubrics". You can roll the same idea in 80 lines.

Run: python 07_evaluation.py    # requires ANTHROPIC_API_KEY
"""

import json
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
# A small "golden set"
# ───────────────────────────────────────────────────────────
DOCS = [
    {"id": "rt", "text": "Acme refunds purchases pro-rata within 30 days. After 30 days, no refunds."},
    {"id": "2fa", "text": "Two-factor authentication is free on all Acme plans."},
    {"id": "sso", "text": "SAML SSO is available on Team and Enterprise plans only."},
    {"id": "rl", "text": "API rate limits: 60 RPM Starter; 600 RPM Team; custom for Enterprise."},
    {"id": "bk", "text": "Backups are nightly with 30-day retention. Point-in-time recovery on Enterprise."},
    {"id": "tr", "text": "Free trial is 14 days, full Team-plan features, no credit card required."},
]

# Each test case has: query, expected docs (gold relevant), expected fact summary.
EVAL_SET = [
    {"query": "What is the refund policy?",
     "expected_docs": ["rt"],
     "must_contain": "30 days"},
    {"query": "Which plans support SSO?",
     "expected_docs": ["sso"],
     "must_contain": "Team"},
    {"query": "Are backups available?",
     "expected_docs": ["bk"],
     "must_contain": "30"},
    {"query": "How long is the free trial?",
     "expected_docs": ["tr"],
     "must_contain": "14"},
    {"query": "Do you accept Bitcoin?",
     "expected_docs": [],          # not in docs — model SHOULD say "I don't know"
     "must_contain": None},
]


# ───────────────────────────────────────────────────────────
# Retriever
# ───────────────────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embedder.encode([d["text"] for d in DOCS], normalize_embeddings=True)


def retrieve(query: str, k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q_emb
    top = np.argsort(scores)[::-1][:k]
    return [DOCS[i] for i in top]


# ───────────────────────────────────────────────────────────
# Answerer
# ───────────────────────────────────────────────────────────
def answer(query: str, retrieved: list[dict]) -> str:
    docs_xml = "\n".join(f"<doc id=\"{d['id']}\">{d['text']}</doc>" for d in retrieved)
    user_msg = (
        f"<documents>\n{docs_xml}\n</documents>\n\n"
        f"Question: {query}"
    )
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=200,
        system=(
            "Answer using ONLY the documents. Cite IDs in [brackets]. "
            "If the documents don't contain the answer, say "
            "\"I don't know based on the provided documents.\""
        ),
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()


# ───────────────────────────────────────────────────────────
# Retrieval metrics: recall@k, precision@k
# ───────────────────────────────────────────────────────────
def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0       # nothing to recall — vacuously satisfied
    hit = len(set(retrieved_ids) & set(expected_ids))
    return hit / len(expected_ids)


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    if not expected_ids:
        return 1.0       # rough: if nothing was relevant, anything we returned is... let's pass through
    hit = len(set(retrieved_ids) & set(expected_ids))
    return hit / len(retrieved_ids)


# ───────────────────────────────────────────────────────────
# Faithfulness via LLM-as-judge
# ───────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You grade RAG answers. Read the QUESTION, the SOURCE DOCUMENTS, and the
ANSWER. Judge:

  faithful (bool): is every claim in the ANSWER supported by the SOURCES?
                   "I don't know" is also faithful when sources lack info.
  relevant (bool): does the ANSWER address the QUESTION?
  cited (bool):    does the ANSWER cite at least one source ID,
                   or correctly say "I don't know"?

Output ONLY a JSON object: {"faithful": bool, "relevant": bool, "cited": bool}
"""


def judge(question: str, docs: list[dict], ans: str) -> dict:
    docs_text = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{docs_text}\n\n"
        f"ANSWER:\n{ans}"
    )
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=120,
        temperature=0,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text.strip()
    text = text.strip("```json").strip("```").strip()
    return json.loads(text)


# ───────────────────────────────────────────────────────────
# Run the evaluation
# ───────────────────────────────────────────────────────────
results = []
for case in EVAL_SET:
    retrieved = retrieve(case["query"], k=3)
    retrieved_ids = [d["id"] for d in retrieved]
    ans = answer(case["query"], retrieved)
    grades = judge(case["query"], retrieved, ans)
    results.append({
        "query": case["query"],
        "retrieved": retrieved_ids,
        "expected": case["expected_docs"],
        "recall@3": recall_at_k(retrieved_ids, case["expected_docs"]),
        "precision@3": precision_at_k(retrieved_ids, case["expected_docs"]),
        **grades,
        "answer": ans,
    })


# ───────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────
print("\n=== per-query results ===")
for r in results:
    print(
        f"\nQ: {r['query']}"
        f"\n  retrieved   : {r['retrieved']}  expected: {r['expected']}"
        f"\n  recall@3={r['recall@3']:.2f}  precision@3={r['precision@3']:.2f}"
        f"\n  faithful={r['faithful']}  relevant={r['relevant']}  cited={r['cited']}"
        f"\n  answer: {r['answer']}"
    )

agg = {
    "recall@3":     sum(r["recall@3"] for r in results) / len(results),
    "precision@3":  sum(r["precision@3"] for r in results) / len(results),
    "faithful%":    sum(r["faithful"] for r in results) / len(results),
    "relevant%":    sum(r["relevant"] for r in results) / len(results),
    "cited%":       sum(r["cited"] for r in results) / len(results),
}
print(f"\n=== aggregate ===")
for k, v in agg.items():
    print(f"  {k:<14} {v:.2f}")


# ───────────────────────────────────────────────────────────
# Tips
# ───────────────────────────────────────────────────────────
# • Build your golden set EARLY. Even 50 questions (with expected facts and
#   relevant doc IDs) catches regressions other metrics miss.
# • Run the eval on EVERY commit that touches retrieval / chunking / prompt.
# • Log per-query results, not just aggregates. The interesting questions are
#   the ones where the model missed.
# • LLM-as-judge has biases — its scores correlate with model quality. Don't
#   judge with the SAME model that's generating the answer (use Sonnet to
#   judge a Haiku app, or vice versa).
# • Frameworks (ragas, TruLens, DeepEval) ship the same recipes plus
#   dashboards and CI integrations. Use them when this homemade harness gets
#   in the way.
