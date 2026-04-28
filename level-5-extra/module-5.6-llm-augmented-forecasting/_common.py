"""Shared helpers for module 5.6: data loaders, LLM clients."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_LEVEL_5 = Path(__file__).resolve().parents[1]
load_dotenv(_LEVEL_5 / ".env")

_M51 = _LEVEL_5 / "module-5.1-financial-timeseries-foundations"
if str(_M51) not in sys.path:
    sys.path.insert(0, str(_M51))

from _common import fetch_market, fetch_fred  # noqa: E402,F401


def anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Default model picks for 2026
CLAUDE_FAST = "claude-haiku-4-5-20251001"     # cheap, fast — sentiment, classification
CLAUDE_SMART = "claude-sonnet-4-6"            # reasoning, agents
GPT_FAST = "gpt-4o-mini"
GPT_SMART = "gpt-4o"
