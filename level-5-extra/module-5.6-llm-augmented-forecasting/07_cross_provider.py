"""
07 — Same prompt, two providers.

Compare Anthropic Claude and OpenAI GPT on the same sentiment-scoring
task. Useful for:
- Sanity-checking against single-vendor lock-in.
- Ensembling LLM outputs.
- Catching cases where one model has a systematic bias.

Run: python 07_cross_provider.py
"""
from __future__ import annotations

import json

from _common import CLAUDE_FAST, GPT_FAST, anthropic_client, openai_client

HEADLINES = [
    "Apple beats Q4 earnings, raises dividend; iPhone sales up 8% YoY",
    "Bitcoin slides 6% after major exchange suspends withdrawals",
    "Nvidia announces next-gen chip; AI demand 'unprecedented'",
    "Treasury yields jump after surprisingly strong jobs report",
]

PROMPT = """Score each headline for short-term (1-5 day) market impact.
Return JSON list with keys: index (int), score (float, -1..+1).
Be conservative — most headlines are 0 or near it."""


def claude_scores() -> list[dict]:
    msg = anthropic_client().messages.create(
        model=CLAUDE_FAST,
        max_tokens=512,
        system=PROMPT,
        messages=[{"role": "user", "content": json.dumps({"headlines": HEADLINES})}],
    )
    text = msg.content[0].text
    return json.loads(text[text.find("["):text.rfind("]") + 1])


def openai_scores() -> list[dict]:
    rsp = openai_client().chat.completions.create(
        model=GPT_FAST,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT + " Wrap the list under key 'scores'."},
            {"role": "user", "content": json.dumps({"headlines": HEADLINES})},
        ],
    )
    obj = json.loads(rsp.choices[0].message.content)
    return obj.get("scores", obj.get("data", []))


if __name__ == "__main__":
    a = claude_scores()
    b = openai_scores()
    print(f"{'idx':<4} {'claude':>8} {'gpt':>8}  headline")
    a_map = {x["index"]: x["score"] for x in a}
    b_map = {x["index"]: x["score"] for x in b}
    for i, h in enumerate(HEADLINES):
        print(f"{i:<4} {a_map.get(i, float('nan')):>+8.2f} {b_map.get(i, float('nan')):>+8.2f}  {h}")

    # Average them as a weak ensemble
    print("\nEnsembled (mean):")
    for i, h in enumerate(HEADLINES):
        avg = 0.5 * (a_map.get(i, 0) + b_map.get(i, 0))
        print(f"  {avg:+.2f}  {h}")
