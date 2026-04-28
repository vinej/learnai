"""
Exercise 4 — Build a golden-set RAG evaluator

A 20+ question evaluation harness for a tiny RAG system. Reports:
  • retrieval recall@3
  • LLM-as-judge faithfulness, relevance, and citation rates

The script asserts the aggregated faithful-and-relevant rate ≥ 80%.

Run: python exercises/04_golden_set_eval.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
from pathlib import Path

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Knowledge base
# ───────────────────────────────────────────────────────────
DOCS = [
    {"id": "rt",   "text": "Acme refunds purchases pro-rata within 30 days. After 30 days, no refunds."},
    {"id": "2fa",  "text": "Two-factor authentication is free on all Acme plans."},
    {"id": "sso",  "text": "SAML SSO is available on Team and Enterprise plans only."},
    {"id": "rl",   "text": "API rate limits: 60 RPM Starter; 600 RPM Team; custom for Enterprise."},
    {"id": "bk",   "text": "Backups are nightly with 30-day retention. Point-in-time recovery on Enterprise."},
    {"id": "tr",   "text": "Free trial is 14 days, full Team-plan features, no credit card required."},
    {"id": "rg",   "text": "Available regions: us-east, us-west, eu-west, ap-southeast."},
    {"id": "sup",  "text": "Standard support is 24/7 chat. Phone support is Enterprise-only."},
    {"id": "cmp",  "text": "Acme is SOC 2 Type II certified, GDPR-ready, HIPAA-eligible on Enterprise."},
    {"id": "auth", "text": "SSO providers: Okta, Azure AD, Google Workspace, generic SAML 2.0."},
]

# 20 questions, each with the relevant doc IDs and a "must contain" snippet.
EVAL_SET = [
    {"q": "What's the refund policy?",                 "docs": ["rt"],   "must": "30 days"},
    {"q": "Is 2FA included on the Starter plan?",      "docs": ["2fa"],  "must": "free"},
    {"q": "Which plans support SSO?",                  "docs": ["sso"],  "must": "Team"},
    {"q": "What is the API rate limit on Team?",       "docs": ["rl"],   "must": "600"},
    {"q": "How long are backups retained?",            "docs": ["bk"],   "must": "30"},
    {"q": "Is point-in-time recovery available?",       "docs": ["bk"],   "must": "Enterprise"},
    {"q": "How many days is the free trial?",          "docs": ["tr"],   "must": "14"},
    {"q": "Do I need a credit card to start a trial?",  "docs": ["tr"],   "must": "no credit card"},
    {"q": "Which AWS regions can I deploy in?",        "docs": ["rg"],   "must": "us-east"},
    {"q": "What support channels are there?",          "docs": ["sup"],  "must": "chat"},
    {"q": "Do you offer phone support?",                "docs": ["sup"],  "must": "Enterprise"},
    {"q": "Are you HIPAA compliant?",                   "docs": ["cmp"],  "must": "HIPAA"},
    {"q": "Can I use Okta for SSO?",                    "docs": ["auth", "sso"],"must": "Okta"},
    {"q": "What's the rate limit for free users?",      "docs": ["rl"],   "must": "60"},
    {"q": "Are you SOC 2 certified?",                   "docs": ["cmp"],  "must": "SOC 2"},
    {"q": "Is two-factor auth optional?",               "docs": ["2fa"],  "must": "free"},
    {"q": "What auth providers do you integrate with?", "docs": ["auth"], "must": "Azure"},
    {"q": "Do you support cryptocurrency payments?",    "docs": [],       "must": None},
    {"q": "What's the latency on the eu-west region?",  "docs": [],       "must": None},
    {"q": "Do you have a Brazilian Portuguese UI?",     "docs": [],       "must": None},
]


# ───────────────────────────────────────────────────────────
# Retriever
# ───────────────────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embedder.encode([d["text"] for d in DOCS], normalize_embeddings=True)


def retrieve(query, k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q_emb
    top = np.argsort(scores)[::-1][:k]
    return [DOCS[i] for i in top]


# ───────────────────────────────────────────────────────────
# Answerer
# ───────────────────────────────────────────────────────────
def answer(query, hits):
    docs_xml = "\n".join(f"<doc id=\"{d['id']}\">{d['text']}</doc>" for d in hits)
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=200,
        system=(
            "Answer using ONLY the documents. Cite IDs in [brackets]. "
            "If the answer isn't in the documents, say "
            "\"I don't know based on the provided documents.\""
        ),
        messages=[{"role": "user",
                   "content": f"<documents>\n{docs_xml}\n</documents>\n\nQuestion: {query}"}],
    )
    return resp.content[0].text.strip()


# ───────────────────────────────────────────────────────────
# Judge
# ───────────────────────────────────────────────────────────
JUDGE_SYS = """\
You grade RAG answers. Output ONLY a compact JSON object with these booleans:
  - faithful: every claim in the answer is supported by the sources
              (treat "I don't know based on the provided documents" as faithful
              when the sources don't contain the answer)
  - relevant: the answer addresses the question (an honest IDK on out-of-scope
              questions counts as relevant)
  - cited:   the answer cites at least one source ID, OR correctly says IDK
"""


def judge(question, docs, ans):
    docs_text = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    user = (f"QUESTION:\n{question}\n\nSOURCES:\n{docs_text}\n\nANSWER:\n{ans}")
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=80,
        temperature=0,
        system=JUDGE_SYS,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ───────────────────────────────────────────────────────────
# Run the eval
# ───────────────────────────────────────────────────────────
results = []
for i, case in enumerate(EVAL_SET):
    hits = retrieve(case["q"], k=3)
    retrieved_ids = [h["id"] for h in hits]
    ans = answer(case["q"], hits)
    grades = judge(case["q"], hits, ans)

    expected = case["docs"]
    recall = (
        len(set(retrieved_ids) & set(expected)) / len(expected)
        if expected else None
    )
    must_pass = (case["must"] is None) or (case["must"].lower() in ans.lower())

    results.append({
        "q": case["q"], "expected": expected, "retrieved": retrieved_ids,
        "recall@3": recall, "must_pass": must_pass, **grades,
    })

# ───────────────────────────────────────────────────────────
# Aggregate
# ───────────────────────────────────────────────────────────
in_scope = [r for r in results if r["expected"]]
out_scope = [r for r in results if not r["expected"]]

agg = {
    "in-scope recall@3":  np.mean([r["recall@3"] for r in in_scope]),
    "must-contain hit %": np.mean([r["must_pass"] for r in results]),
    "faithful %":         np.mean([r["faithful"] for r in results]),
    "relevant %":         np.mean([r["relevant"] for r in results]),
    "cited %":            np.mean([r["cited"] for r in results]),
}

print(f"\n{'metric':<25}  value")
for k, v in agg.items():
    print(f"  {k:<25}  {v:.3f}")

faithful_relevant = np.mean([r["faithful"] and r["relevant"] for r in results])
print(f"\n→ faithful-AND-relevant rate: {faithful_relevant:.3f}")

assert faithful_relevant >= 0.80, (
    f"target ≥ 0.80, got {faithful_relevant:.2f}. Inspect the per-question "
    "results to find regressions."
)
print("eval suite passes ✅")


# ── Workflow ────────────────────────────────────────────────────────────────
# This file IS your CI test for the RAG system. Run it on every PR that
# touches retrieval, chunking, or prompts. Track the aggregate scores over
# time. Aim to expand the eval set whenever a real user question reveals
# a gap.
# ─────────────────────────────────────────────────────────────────────────────
