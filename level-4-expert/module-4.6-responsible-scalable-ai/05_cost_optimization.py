"""
05 — Cost and latency optimization

LLM apps' per-request cost can drift 100× depending on engineering choices.
The big knobs:

  RIGHT-SIZING
    Use the SMALLEST model that hits your quality bar. Test Haiku first.
    Reserve Opus for hard cases.

  MODEL CASCADE / ROUTING
    Cheap model first; route to bigger model only on low-confidence cases.
    Often saves 80-90% of inference cost with little quality loss.

  PROMPT CACHING (Module 4.1.06)
    For ANY repeated long prompt — system instructions, RAG context, tool
    definitions. ~90% input cost reduction once cached.

  SEMANTIC CACHE
    For PRODUCT-LEVEL repetition: similar user queries return the same
    answer. Use embeddings + a vector store as a "memoization" layer.

  RESPONSE CACHE
    Exact-match cache on (prompt → response). Cheap, easy, useful for
    deterministic prompts.

  OUTPUT BUDGETING
    Cap max_tokens. Models tend to fill the budget if it's loose.

  BATCHING (when latency permits)
    Send N independent prompts in one batch. Anthropic Batch API offers
    50% off for non-real-time workloads.

  DISTILLATION
    Train a small fine-tuned model on big-model outputs. Replace 90% of
    Opus calls with a self-hosted Haiku-class model.

This file demonstrates a model cascade and a semantic cache.

Run: python 05_cost_optimization.py    # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import cost_of_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Model cascade
# ───────────────────────────────────────────────────────────
# 1. Try Haiku.
# 2. If Haiku says "I'm not confident" (or fails a quality check), retry
#    with Sonnet.
SMALL_MODEL = "claude-haiku-4-5-20251001"
BIG_MODEL   = "claude-sonnet-4-6"


def ask_with_cascade(question: str) -> dict:
    """Try Haiku; only escalate when needed."""
    # First pass: Haiku, asked to flag uncertainty.
    haiku = client.messages.create(
        model=SMALL_MODEL,
        max_tokens=300,
        system=(
            "Answer the user's question. If you're NOT confident or it requires "
            "complex reasoning, output exactly: ESCALATE. Otherwise give the "
            "answer in 1-3 sentences."
        ),
        messages=[{"role": "user", "content": question}],
    )
    haiku_text = haiku.content[0].text.strip()
    haiku_cost = cost_of_usage(haiku.usage, SMALL_MODEL)

    if "ESCALATE" in haiku_text.upper():
        sonnet = client.messages.create(
            model=BIG_MODEL,
            max_tokens=500,
            system="Answer the user's question carefully and concisely.",
            messages=[{"role": "user", "content": question}],
        )
        return {
            "answer": sonnet.content[0].text.strip(),
            "model_used": BIG_MODEL,
            "cost": haiku_cost + cost_of_usage(sonnet.usage, BIG_MODEL),
            "escalated": True,
        }

    return {
        "answer": haiku_text,
        "model_used": SMALL_MODEL,
        "cost": haiku_cost,
        "escalated": False,
    }


print("=== model cascade demo ===")
QUESTIONS = [
    "What's the capital of Japan?",
    "Explain how the cumulants of a probability distribution relate to its moments, with worked formulas for the first three.",
    "What's 2 + 2?",
    "Why did the Roman Empire's Western half fall while the Eastern half persisted for another thousand years?",
]

total_cost = 0.0
for q in QUESTIONS:
    out = ask_with_cascade(q)
    total_cost += out["cost"]
    badge = "BIG" if out["escalated"] else "small"
    print(f"  [{badge}] ${out['cost']:.5f}  {q[:60]}")
    print(f"      → {out['answer'][:90]}...")
print(f"\ntotal: ${total_cost:.5f}")


# ───────────────────────────────────────────────────────────
# Semantic cache
# ───────────────────────────────────────────────────────────
# Embed each query; if a previous query is near (cosine ≥ T), reuse the answer.
class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.queries: list[str] = []
        self.embs: list[np.ndarray] = []
        self.answers: list[str] = []

    def _embed(self, text: str) -> np.ndarray:
        # Hash-based pseudo-embedding to keep this demo dep-light.
        # In production, use sentence-transformers or a provider embedding.
        rng = np.random.default_rng(hash(text) & 0xffffffff)
        v = rng.standard_normal(64)
        return v / np.linalg.norm(v)

    def get(self, query: str) -> str | None:
        if not self.queries:
            return None
        q = self._embed(query)
        sims = np.array([float(q @ e) for e in self.embs])
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            return self.answers[i]
        return None

    def put(self, query: str, answer: str) -> None:
        self.queries.append(query)
        self.embs.append(self._embed(query))
        self.answers.append(answer)


print("\n=== semantic cache shape ===")
print("Pseudocode: embed query → check cache → on hit return cached answer; "
      "on miss, run the model and put(...) the result. Use a real embedder "
      "(sentence-transformers / Voyage / OpenAI) and a vector store at scale.")


# ───────────────────────────────────────────────────────────
# Where the savings really compound
# ───────────────────────────────────────────────────────────
COMPOUNDING = """\
WALKING DOWN THE COST LADDER FOR A CHATBOT

  Baseline:     Opus, 5k system, 1k user, 500 reply, 100 calls.
                ≈ $0.46 / call, $46 / day.

  Step 1: switch to Haiku (4× cheaper input, 5× cheaper output).
          $0.06 / call ↓ $6 / day.

  Step 2: prompt-cache the 5k system prompt (reads ~10% of normal).
          $0.025 / call ↓ $2.5 / day.

  Step 3: semantic cache (50% hit rate at zero cost).
          $0.013 / call ↓ $1.3 / day.

  Step 4: cascade: 80% answered by Haiku, 20% escalated to Sonnet.
          $0.013 + 0.20 * 0.10 = $0.033 / call ... but we only escalate
          on hard queries, so AVERAGE quality is preserved.

Net: ~$1.3 / day vs $46 / day at the start. Same product surface.
"""
print(COMPOUNDING)


# ───────────────────────────────────────────────────────────
# Latency rules of thumb
# ───────────────────────────────────────────────────────────
LATENCY = """\
LATENCY RULES (Anthropic API, rough):
  - First-token latency:   ~200-400ms once warmed up.
  - Decode rate:            ~50-200 tokens/sec for Haiku, less for Opus.
  - Cached prompts skip prefill — the dominant savings on long prompts.
  - Streaming hides latency: user sees first token in ~300ms regardless of
    the total length.
  - Smaller model = lower TTFT and faster decode.

FAST PATHS for user-perceived latency:
  - Stream the response (Module 4.1.03).
  - Cache aggressively (file 05_cost_optimization).
  - Cascade to a small model for the easy 80%.
  - Pre-compute deterministic parts (templates, embeddings) at index time.
"""
print(LATENCY)
