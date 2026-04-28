"""Generate a short, calibrated narrative for a forecast bundle."""
from __future__ import annotations

import json

from forecaster.config import SETTINGS

PROMPT = """You write clear, calibrated narratives for short-horizon market forecasts.

Rules:
- Lead with direction and confidence (avoid certainty).
- State the interval as a range, not a single number.
- Mention 1-2 model agreements/disagreements.
- Mention 1 named risk that could invalidate the forecast.
- Maximum 5 sentences. Never recommend trades.
"""


def narrate(forecast: dict) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=SETTINGS.anthropic_key)
    msg = client.messages.create(
        model=SETTINGS.claude_fast,
        max_tokens=400,
        system=PROMPT,
        messages=[{"role": "user", "content": json.dumps(forecast, indent=2)}],
    )
    return msg.content[0].text
