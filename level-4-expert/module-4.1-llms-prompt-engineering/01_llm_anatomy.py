"""
01 — LLM anatomy (engineer's view)

Modern LLMs are DECODER-ONLY transformers (you built one in Module 3.3.05).
The recipe:

  1. Tokenize the prompt → list of integer token ids.
  2. Compute logits for the NEXT token: logits = model(input_ids).
  3. Sample one token from softmax(logits / T).
  4. Append it to input_ids. Repeat steps 2-4 until EOS or max_tokens.

That's autoregressive generation. Training the model is just the same loop
in reverse: given a corpus, predict each next token.

Why models that just predict the next token end up able to write code,
reason, and converse is — genuinely — one of the most surprising results
in CS in the last decade.

This file demonstrates the core ideas with a synthetic "language model"
(no real model, no API call) so you can run it offline.

Run: python 01_llm_anatomy.py
"""

import numpy as np


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# A toy "language model": next-token distribution given context
# ───────────────────────────────────────────────────────────
# In a real LLM, this is a giant transformer. Here we use a 4-gram lookup
# table built from a tiny corpus, just to make the loop concrete.
CORPUS = (
    "the cat sat on the mat. "
    "the cat ran on the rug. "
    "a cat and a dog sat on the mat. "
    "the dog ran on the rug. "
    "the cat sat with the dog. "
)


def build_ngram(text: str, n: int = 3) -> dict:
    """Map (last n-1 tokens) → counter over next tokens."""
    tokens = text.split()
    table = {}
    for i in range(len(tokens) - n + 1):
        ctx = tuple(tokens[i:i + n - 1])
        nxt = tokens[i + n - 1]
        table.setdefault(ctx, {}).setdefault(nxt, 0)
        table[ctx][nxt] += 1
    return table


table = build_ngram(CORPUS)
print(f"unique 2-token contexts: {len(table)}")


# ───────────────────────────────────────────────────────────
# Get next-token "logits" (here: counts) for a context
# ───────────────────────────────────────────────────────────
def next_token_logits(ctx: tuple) -> dict[str, float]:
    """Return next-token "logits" — a dict of token → log-count."""
    counts = table.get(ctx, {})
    if not counts:
        # Backoff to bigram
        for shorter in [(ctx[-1],) if ctx else ()]:
            if shorter in table:
                counts = table[shorter]
                break
    return {tok: float(np.log(c)) for tok, c in counts.items()}


# ───────────────────────────────────────────────────────────
# The generation loop
# ───────────────────────────────────────────────────────────
def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    z = np.array(logits) / max(temperature, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).tolist()


def generate(prompt: str, max_new: int = 10, temperature: float = 1.0) -> str:
    tokens = prompt.split()
    for _ in range(max_new):
        ctx = tuple(tokens[-2:])
        logits = next_token_logits(ctx)
        if not logits:
            break
        toks = list(logits)
        probs = softmax(list(logits.values()), temperature=temperature)
        nxt = rng.choice(toks, p=probs)
        tokens.append(nxt)
        if nxt.endswith("."):
            break
    return " ".join(tokens)


print("\nrunning the autoregressive loop on a toy 4-gram model:")
for _ in range(3):
    print(" ", generate("the cat", max_new=10, temperature=0.8))


# ───────────────────────────────────────────────────────────
# What's different in a real LLM
# ───────────────────────────────────────────────────────────
# • The "model" is a giant transformer (billions of parameters), not a
#   3-token lookup table. Same forward-pass-then-sample loop.
# • Tokenization uses subword BPE/SentencePiece (Module 3.3.01), not whitespace.
# • The KV CACHE — at each step, the model caches keys/values from previous
#   tokens so generation is O(N) instead of O(N²). That's why the cost of
#   the FIRST token (prefill) is much higher than subsequent tokens (decode).
# • Speculative decoding, draft models, batching, paged attention — all
#   tricks to make decode faster.
# • EOS, stop sequences, max_tokens — three ways to halt generation.


# ───────────────────────────────────────────────────────────
# Concepts you'll meet in the next files
# ───────────────────────────────────────────────────────────
# • TEMPERATURE / TOP-K / TOP-P  — how to pick `nxt` (file 02).
# • CONTEXT WINDOW                — max input length the model can handle.
# • TOKEN PRICING                 — input vs output per million tokens.
# • PROMPTING PATTERNS            — system / user / assistant turns,
#                                   few-shot, chain-of-thought (file 04).
# • TOOL USE                      — model emits structured calls to
#                                   functions you implement (file 07).
