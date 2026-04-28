"""
03 — Structured extraction from earnings text.

Goal: turn unstructured earnings release / call text into a list of
records:
    [{metric: "revenue", value: 89.5, unit: "billion USD",
      period: "Q4 FY24", change: "+8% YoY", source_phrase: "..."}]

Use Claude with the `tool_use` API (see L4.1 / L4.4) for guaranteed
structured output. Helpful for:
- Building a clean panel of guidance vs actuals.
- Anomaly detection in releases.
- Rapid scenario construction.

Run: python 03_earnings_extraction.py
"""
from __future__ import annotations

import json

from _common import CLAUDE_SMART, anthropic_client

SAMPLE = """
Apple today announced financial results for its fiscal 2024 fourth quarter
ended September 28, 2024. The Company posted quarterly revenue of $94.9
billion, up 6 percent year over year, and quarterly diluted earnings per
share of $1.64. Services revenue reached an all-time high of $24.97 billion,
up 12% year over year. iPhone revenue was $46.22 billion, up 6% year over
year. The Board of Directors has declared a cash dividend of $0.25 per share,
payable on November 14, 2024.
"""

EXTRACT_TOOL = {
    "name": "record_metrics",
    "description": "Record one or more financial metrics extracted from the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                        "period": {"type": "string"},
                        "yoy_change_pct": {"type": ["number", "null"]},
                        "source_phrase": {"type": "string"},
                    },
                    "required": ["metric", "value", "unit", "period", "source_phrase"],
                }
            }
        },
        "required": ["metrics"],
    },
}


def extract(text: str) -> list[dict]:
    client = anthropic_client()
    msg = client.messages.create(
        model=CLAUDE_SMART,
        max_tokens=2048,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_metrics"},
        messages=[{"role": "user", "content":
                   "Extract every financial metric you find. Be precise.\n\n" + text}],
    )
    for block in msg.content:
        if block.type == "tool_use" and block.name == "record_metrics":
            return block.input["metrics"]
    return []


if __name__ == "__main__":
    metrics = extract(SAMPLE)
    print(json.dumps(metrics, indent=2))
