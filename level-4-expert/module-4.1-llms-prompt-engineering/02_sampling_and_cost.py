"""
02 — Sampling and cost

SAMPLING controls HOW the next token is picked from the model's distribution.

  TEMPERATURE T   — divides logits before softmax. T=0 deterministic
                    (always argmax); T=1 unchanged; T>1 more random.
  TOP-K           — keep only the k highest-probability tokens, renormalize.
  TOP-P (NUCLEUS) — keep the smallest set of tokens whose cumulative
                    probability exceeds p. Adapts to flat vs peaked dists.
  REPETITION PEN  — penalize tokens already produced, discourages loops.

COST. LLMs charge per million input + output tokens. Long inputs and long
outputs both cost. Short, focused prompts and small max_tokens save money.

Run: python 02_sampling_and_cost.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# A made-up next-token distribution
# ───────────────────────────────────────────────────────────
tokens = ["the", "a", "and", "of", "to", "in", "is", "that", "it", "with",
          "for", "be", "on", "as", "this"]
logits = np.array([4.5, 4.2, 3.0, 2.8, 2.5, 2.3, 2.0, 1.8, 1.5, 1.2,
                   1.0, 0.8, 0.6, 0.5, 0.3])


# ───────────────────────────────────────────────────────────
# Sampling functions
# ───────────────────────────────────────────────────────────
def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / max(temperature, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top_k(probs: np.ndarray, k: int) -> np.ndarray:
    keep = np.argpartition(probs, -k)[-k:]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    return masked / masked.sum()


def top_p(probs: np.ndarray, p: float) -> np.ndarray:
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = (cumulative <= p).sum() + 1
    keep_idx = order[:cutoff]
    masked = np.zeros_like(probs)
    masked[keep_idx] = probs[keep_idx]
    return masked / masked.sum()


def sample(probs: np.ndarray, n: int = 1000) -> dict[str, int]:
    """Sample tokens from the distribution and tally counts."""
    out = rng.choice(len(probs), size=n, p=probs)
    return {tokens[i]: int((out == i).sum()) for i in range(len(probs))}


# ───────────────────────────────────────────────────────────
# Compare temperature settings
# ───────────────────────────────────────────────────────────
configs = {
    "T=0.3 (greedy-ish)": softmax(logits, temperature=0.3),
    "T=1.0 (default)":    softmax(logits, temperature=1.0),
    "T=1.5 (creative)":   softmax(logits, temperature=1.5),
    "T=1.0 + top_k=3":    top_k(softmax(logits, 1.0), k=3),
    "T=1.0 + top_p=0.7":  top_p(softmax(logits, 1.0), p=0.7),
}

fig, axes = plt.subplots(1, len(configs), figsize=(16, 4))
for ax, (name, probs) in zip(axes, configs.items()):
    ax.bar(tokens, probs, color="steelblue")
    ax.set_title(name, fontsize=10)
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylim(0, 1)
fig.suptitle("Effect of sampling parameters on the SAME logits")
fig.tight_layout()

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "02_sampling.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / '02_sampling.png'}")


# ───────────────────────────────────────────────────────────
# Picking sampling values
# ───────────────────────────────────────────────────────────
# Reasoning, code, structured output       → temperature ≈ 0 (deterministic).
# General assistant / Q&A                  → 0.7-1.0 (default).
# Brainstorming, writing, creative tasks   → 0.9-1.2.
# Avoid temperature > 1.5 — output usually degrades to nonsense.
#
# top_k or top_p are usually applied IN ADDITION to temperature. Most APIs
# expose ONE of them (Anthropic uses top_p; OpenAI uses both). Default values
# are usually sensible; only override if you know why.


# ───────────────────────────────────────────────────────────
# Token & cost accounting
# ───────────────────────────────────────────────────────────
# Pricing is per MILLION tokens. Both INPUT and OUTPUT are billed
# (output is usually 4-5× more expensive than input).
#
# Approximate Claude Haiku-class pricing (check console.anthropic.com for current):
PRICING = {
    "haiku":  {"input": 0.80,  "output": 4.00},     # USD per 1M tokens, as a rough number
    "sonnet": {"input": 3.00,  "output": 15.00},
    "opus":   {"input": 15.00, "output": 75.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# How much for 100 chats with a 5k-token system prompt + 1k user + 500 reply?
input_per_call  = 5000 + 1000
output_per_call = 500
n_calls = 100

print("\n=== rough cost: 100 chat calls (5k system + 1k user, 500 reply) ===")
for model in PRICING:
    cost = estimate_cost(model, input_per_call * n_calls, output_per_call * n_calls)
    print(f"  {model:<7}  ${cost:.2f}")


# ───────────────────────────────────────────────────────────
# Where the savings come from (preview)
# ───────────────────────────────────────────────────────────
# • PROMPT CACHING — cached input tokens cost ~10% of normal. If your 5k
#   system prompt is the same across 100 calls, you'll save 90% on input.
# • SHORT OUTPUTS — set max_tokens to what you actually need. The model
#   tends to fill the budget if you give it too much.
# • SMALLER MODEL — Haiku-class models are 5-25× cheaper than Opus. Try
#   them first; only escalate when you see Haiku fail.
#
# File 06 covers prompt caching in detail.


# ───────────────────────────────────────────────────────────
# Counting tokens
# ───────────────────────────────────────────────────────────
# Anthropic's tokenizer is not directly distributed. Two ways to count:
#   1. Send a `messages.count_tokens(...)` API call (cheap, exact).
#   2. Approximate offline: ~4 chars/token for English. Fine for budgeting.

def approx_tokens(text: str) -> int:
    """Rough heuristic: 1 token ≈ 4 chars of English text."""
    return max(1, len(text) // 4)


sample = "Hello, world! This is a sample sentence to estimate tokens."
print(f"\napprox tokens for '{sample}': ~{approx_tokens(sample)}")
