"""
06 — Forecast narrative.

Take the numerical output of an ensemble (point + interval + per-model
contributions) and produce a clean explanation for a non-technical reader.

This is where LLMs genuinely earn their keep in forecasting:
the numbers are the numbers, but a good narrative converts them into
something a stakeholder can act on.

Run: python 06_forecast_narrative.py
"""
from __future__ import annotations

import json

from _common import CLAUDE_FAST, anthropic_client

PROMPT = """You write clear, calibrated narratives for short-horizon market forecasts.

Rules:
- Lead with the headline: direction and confidence.
- Quote the interval as a "between X and Y" range, not a single number.
- Mention 1-2 model agreements/disagreements that matter.
- Mention 1 named risk that could invalidate the forecast.
- Maximum 5 sentences. No hedging beyond what the numbers support.
- Never claim certainty. Never recommend trades.
"""


def narrate(forecast: dict) -> str:
    msg = anthropic_client().messages.create(
        model=CLAUDE_FAST,
        max_tokens=400,
        system=PROMPT,
        messages=[{"role": "user", "content": json.dumps(forecast, indent=2)}],
    )
    return msg.content[0].text


if __name__ == "__main__":
    fc = {
        "ticker": "SPY",
        "horizon_days": 21,
        "as_of": "2026-04-28",
        "point_forecast_pct": 0.6,
        "interval_pct": [-3.4, 4.7],
        "interval_level": 0.8,
        "model_contributions_pct": {
            "lightgbm": 0.4,
            "patchtst": 0.7,
            "chronos_bolt": 0.5,
            "ets": 0.9,
            "llm_view": 0.4,
        },
        "macro_notes": "10Y yield at 4.32%, no Fed meeting in window, CPI release on day 7.",
    }
    print(narrate(fc))
