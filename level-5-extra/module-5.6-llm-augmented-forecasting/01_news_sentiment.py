"""
01 — News sentiment via Claude Haiku.

We score a list of headlines on a -1 (very bearish) to +1 (very bullish)
scale, with structured output. Cheap model — Haiku is enough for this.

For real production use you'd:
- Pull headlines from a real feed (Polygon, Benzinga, Refinitiv, RSS).
- Normalize the timestamps to "before/after market close".
- Aggregate to a daily score per ticker (mean, with weighting by source).

Run: python 01_news_sentiment.py
"""
from __future__ import annotations

import json
from typing import Any

from _common import CLAUDE_FAST, anthropic_client

HEADLINES = [
    "Apple beats Q4 earnings, raises dividend; iPhone sales up 8% YoY",
    "Fed signals possible rate cut in December as inflation cools",
    "Bitcoin slides 6% after major exchange suspends withdrawals",
    "Tesla recalls 200k vehicles over autopilot software flaw",
    "Nvidia announces next-gen chip; AI demand 'unprecedented'",
    "Oil falls 3% on weaker China manufacturing PMI",
    "Treasury yields jump after surprisingly strong jobs report",
]


SYSTEM = """You score financial headlines for short-term (1-5 day) market impact.
Return strict JSON: a list of objects with keys
  index   (int, position in input)
  asset   (string, the most-impacted asset class — equities | crypto | rates | commodities | fx | mixed)
  score   (float, -1.0 to +1.0; negative = bearish for the asset, positive = bullish)
  reason  (string, <= 12 words)
"""


def score_headlines(headlines: list[str]) -> list[dict[str, Any]]:
    client = anthropic_client()
    msg = client.messages.create(
        model=CLAUDE_FAST,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": json.dumps({"headlines": headlines}),
        }],
    )
    text = msg.content[0].text
    # Find the JSON list
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])


if __name__ == "__main__":
    out = score_headlines(HEADLINES)
    for h, row in zip(HEADLINES, out):
        print(f"{row['score']:+.2f} [{row['asset']:<11}] {h}")
        print(f"           reason: {row['reason']}")
