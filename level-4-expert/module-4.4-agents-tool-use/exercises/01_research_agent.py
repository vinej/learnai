"""
Exercise 1 — Research agent with citations

A multi-step agent that:
  1. Takes a research question.
  2. Searches a (mocked) knowledge base via a `web_search` tool.
  3. Reads the most relevant article via a `fetch_url` tool.
  4. Synthesizes a short answer with [citations].

The script asserts the answer cites at least one source AND contains a
domain-specific keyword from the gold answer.

Run: python exercises/01_research_agent.py    # requires ANTHROPIC_API_KEY
"""

import json
import re
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# A mock "internet"
# ───────────────────────────────────────────────────────────
ARTICLES = {
    "transformer-attention": {
        "title": "Attention is All You Need",
        "body": (
            "The transformer architecture, introduced in 2017, replaced "
            "recurrence with self-attention. Self-attention computes "
            "weighted combinations of all positions in the input sequence "
            "via scaled dot-product attention over learned queries, keys, "
            "and values."
        ),
    },
    "lstm-vanishing": {
        "title": "Why LSTMs Lost to Transformers",
        "body": (
            "Despite gating, LSTMs train sequentially and struggle with "
            "very long-range dependencies because gradients still attenuate "
            "across hundreds of steps. Transformers process the whole "
            "sequence in parallel and attend to any position directly."
        ),
    },
    "rope-positional": {
        "title": "Rotary Positional Embeddings",
        "body": (
            "RoPE (Rotary Position Embedding) injects positional information "
            "by rotating query and key vectors in 2-D subspaces by an angle "
            "proportional to the position. It generalizes to longer sequences "
            "than seen at training time and powers modern LLMs like Llama."
        ),
    },
    "rlhf-overview": {
        "title": "RLHF in a Nutshell",
        "body": (
            "Reinforcement Learning from Human Feedback aligns an LLM by "
            "training a reward model on human preference pairs, then "
            "fine-tuning the policy with PPO to maximize that reward. "
            "Modern alternatives include DPO, KTO, and ORPO."
        ),
    },
}


def web_search(query: str, k: int = 3) -> str:
    q = re.findall(r"\w+", query.lower())
    scored = []
    for url, art in ARTICLES.items():
        score = sum(1 for w in q if w in (url + " " + art["title"] + " " + art["body"]).lower())
        if score:
            scored.append({"url": url, "title": art["title"], "score": score})
    scored.sort(key=lambda x: -x["score"])
    return json.dumps({"results": scored[:k]})


def fetch_url(url: str) -> str:
    art = ARTICLES.get(url)
    if not art:
        return json.dumps({"error": "not found"})
    return json.dumps({"url": url, "title": art["title"], "body": art["body"]})


TOOLS = [
    {
        "name": "web_search",
        "description": "Search for relevant pages. Returns a JSON list of {url, title, score}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k":     {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch the body of a page by URL slug.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]

LOCAL = {"web_search": web_search, "fetch_url": fetch_url}


SYSTEM = """\
You are a research assistant. To answer the user, you MUST:
  1. Use web_search to find relevant pages.
  2. Use fetch_url to read at least one promising page.
  3. Write a short answer (≤ 4 sentences) citing the page URL slug in
     square brackets, e.g. [transformer-attention].
  4. If the question is unanswerable from the available pages, say
     "I don't know based on what I could find."
"""


def run_agent(question: str, max_iter: int = 6) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_iter):
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=600,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  → {b.name}({b.input})")
                    out = LOCAL[b.name](**b.input)
                    print(f"    {out[:120]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue
        return f"(unexpected stop: {resp.stop_reason})"
    return "(max iterations)"


# ───────────────────────────────────────────────────────────
# Question + assertions
# ───────────────────────────────────────────────────────────
QUESTION = "Why did the transformer architecture replace LSTMs in NLP?"
print(f"Q: {QUESTION}\n")
ans = run_agent(QUESTION)
print(f"\n=== answer ===\n{ans}")

# Assertions
assert re.search(r"\[[a-z0-9-]+\]", ans), \
    "answer must cite at least one source in [brackets]"
assert "transformer" in ans.lower() or "attention" in ans.lower(), \
    "answer should mention key concepts"
print("\nresearch agent works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add a `summarize_page` tool to compress long pages before reading.
# • Add a 5-step max + tool budget. Test with intentionally long-running
#   queries and confirm the budget triggers cleanly.
# • Replace the mock with a real Tavily / Brave Search API.
# • Add a critic agent that verifies citations actually support the claims.
# ─────────────────────────────────────────────────────────────────────────────
