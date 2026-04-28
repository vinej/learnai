"""
05 — A 3-agent forecast loop (analyst + skeptic + forecaster).

Pattern:
- ANALYST: given (recent context, numerical baseline forecast, news),
  proposes a directional view + reasoning.
- SKEPTIC: argues the strongest case AGAINST the analyst.
- FORECASTER: reconciles the two and outputs a final point + interval.

This is mostly a way to surface assumptions and stress-test a forecast.
It will rarely beat a tuned numerical model on accuracy, but the trace
is useful in production: it tells you WHY the forecast is what it is.

Run: python 05_multi_agent.py
"""
from __future__ import annotations

import json

from _common import CLAUDE_SMART, anthropic_client


def call(role_system: str, user: str) -> str:
    msg = anthropic_client().messages.create(
        model=CLAUDE_SMART,
        max_tokens=1024,
        system=role_system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


ANALYST = """You are a buy-side analyst. Given a numerical baseline forecast and
context, propose a directional view and the 3 key drivers. Be specific."""

SKEPTIC = """You are a portfolio risk officer. Read the analyst's view and argue
the strongest case AGAINST it. Identify what could make this forecast wrong."""

FORECASTER = """You are the final decision-maker. Given the baseline forecast,
the analyst's view, and the skeptic's rebuttal, produce a final
short-horizon forecast as JSON:
{ "direction": "up"|"down"|"flat", "confidence": 0..1,
  "expected_move_bps": int, "key_risk": "..." }"""


def run(context: str, baseline: dict) -> dict:
    user_for_analyst = f"Context:\n{context}\n\nBaseline forecast: {json.dumps(baseline)}"
    analyst_view = call(ANALYST, user_for_analyst)
    print("\n--- ANALYST ---\n", analyst_view)

    skeptic_view = call(SKEPTIC, f"Analyst said:\n{analyst_view}")
    print("\n--- SKEPTIC ---\n", skeptic_view)

    user_for_fc = (f"Baseline: {json.dumps(baseline)}\n\n"
                    f"Analyst:\n{analyst_view}\n\nSkeptic:\n{skeptic_view}")
    raw = call(FORECASTER, user_for_fc)
    print("\n--- FORECASTER ---\n", raw)

    start, end = raw.find("{"), raw.rfind("}") + 1
    return json.loads(raw[start:end])


if __name__ == "__main__":
    context = ("SPY closed at 525, down 1.4% on the day. 10Y yield at 4.32%, "
                "+8bps. CPI release tomorrow at 8:30 ET; consensus 3.0% YoY. "
                "VIX at 17. Last week's news cycle dominated by Fed minutes "
                "showing a bias toward holding rates higher for longer.")
    baseline = {"horizon_days": 5, "expected_logret": -0.001, "p10": -0.025, "p90": 0.022}
    final = run(context, baseline)
    print("\n--- FINAL ---")
    print(json.dumps(final, indent=2))
