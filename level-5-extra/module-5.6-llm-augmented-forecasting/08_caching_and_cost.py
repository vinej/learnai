"""
08 — Prompt caching for the recurring "system + history" block.

A typical sentiment scorer sends the same long system prompt and the
same recent-news context block hundreds of times per day, with only
a small per-call user message changing. Anthropic's prompt caching
(L4.1) lets you mark stable prefixes for caching, dropping cost by
50-90% on those tokens.

This file shows the API and measures the savings.

Run: python 08_caching_and_cost.py
"""
from __future__ import annotations

import time

from _common import CLAUDE_FAST, anthropic_client

LONG_SYSTEM = """You are a financial news classifier.

You will see a JSON object containing one headline. You must output a
single JSON object with keys:
  asset_class : equities | crypto | rates | commodities | fx | mixed
  direction   : bullish | bearish | neutral
  magnitude   : 0..1 (subjective short-horizon market move expected)
  reason      : <= 12 words

Rules:
- Be conservative. Most macro headlines are neutral.
- Direction must reflect the FIRST-ORDER market interpretation, not your view.
- For headlines that could move multiple asset classes, pick the largest one.
- Never invent specific numbers.
- Never recommend trades.

Examples and edge cases follow:

Example 1:
  Input: { "headline": "Apple beats Q4 earnings, raises dividend" }
  Output: { "asset_class": "equities", "direction": "bullish", "magnitude": 0.4, "reason": "earnings beat with capital return" }

Example 2:
  Input: { "headline": "Bitcoin slides 6% after exchange suspends withdrawals" }
  Output: { "asset_class": "crypto", "direction": "bearish", "magnitude": 0.7, "reason": "infrastructure stress on top exchange" }

(... imagine 50 more lines of style guidance and edge cases ...)
""" * 4   # pad to make caching worthwhile (~5k tokens)


HEADLINES = [
    "Fed signals possible rate cut in December as inflation cools",
    "Tesla recalls 200k vehicles over autopilot software flaw",
    "Oil falls 3% on weaker China manufacturing PMI",
    "Treasury yields jump after surprisingly strong jobs report",
    "Microsoft and OpenAI announce new partnership terms",
] * 4   # repeat to amortize cache


def score_with_caching(headline: str) -> dict:
    client = anthropic_client()
    msg = client.messages.create(
        model=CLAUDE_FAST,
        max_tokens=200,
        system=[
            {"type": "text", "text": LONG_SYSTEM, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": f'{{"headline": "{headline}"}}'}],
    )
    return {
        "input_tokens": msg.usage.input_tokens,
        "cache_creation": getattr(msg.usage, "cache_creation_input_tokens", 0),
        "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0),
        "output_tokens": msg.usage.output_tokens,
        "text": msg.content[0].text,
    }


if __name__ == "__main__":
    start = time.time()
    total_input = 0
    total_creation = 0
    total_read = 0
    for h in HEADLINES:
        r = score_with_caching(h)
        total_input += r["input_tokens"]
        total_creation += r["cache_creation"]
        total_read += r["cache_read"]
    dt = time.time() - start

    # Anthropic pricing (Apr 2026 — verify live):
    # Cache writes are 1.25x base price, cache reads are 0.1x base price.
    # base = $0.80 / Mtok input for Haiku 4.5
    BASE = 0.80
    cost_uncached = (total_input + total_creation + total_read) / 1e6 * BASE
    cost_cached = (total_input / 1e6 * BASE
                    + total_creation / 1e6 * BASE * 1.25
                    + total_read / 1e6 * BASE * 0.10)
    print(f"\nWall time      : {dt:.1f}s")
    print(f"Input tokens   : {total_input:,}")
    print(f"Cache write    : {total_creation:,}")
    print(f"Cache read     : {total_read:,}")
    print(f"\nApprox cost without caching: ${cost_uncached:.4f}")
    print(f"Approx cost with caching   : ${cost_cached:.4f}")
    print(f"Savings                    : {(1 - cost_cached / cost_uncached) * 100:.1f}%")
