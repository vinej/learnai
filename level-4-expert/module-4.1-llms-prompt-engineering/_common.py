"""
Shared helpers used by API-calling lessons in Module 4.1.

Put here to keep individual files focused on the lesson.
"""

import os
import sys


# Default model — Haiku is cheapest and fast enough for learning.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def require_api_key() -> None:
    """Print a friendly message and exit if the API key isn't set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "\n=== ANTHROPIC_API_KEY not set ===\n"
            "Set it in your shell, then re-run:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'   # Windows PowerShell\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'    # macOS/Linux\n",
            file=sys.stderr,
        )
        sys.exit(1)


def cost_of_usage(usage, model: str = DEFAULT_MODEL) -> float:
    """Rough cost in USD from a `response.usage` object.

    Approximate per-million-token rates. Check console.anthropic.com for current.
    """
    rates = {
        "claude-haiku-4-5-20251001":  {"in": 0.80,  "out": 4.00,
                                         "cache_write": 1.00, "cache_read": 0.08},
        "claude-sonnet-4-6":          {"in": 3.00,  "out": 15.00,
                                         "cache_write": 3.75, "cache_read": 0.30},
        "claude-opus-4-7":            {"in": 15.00, "out": 75.00,
                                         "cache_write": 18.75, "cache_read": 1.50},
    }
    r = rates.get(model, rates[DEFAULT_MODEL])
    cost = (
        (usage.input_tokens or 0) * r["in"]
        + (usage.output_tokens or 0) * r["out"]
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * r["cache_write"]
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * r["cache_read"]
    ) / 1_000_000
    return cost


def print_usage(label: str, usage, model: str = DEFAULT_MODEL) -> None:
    print(
        f"\n[{label}] usage: "
        f"input={usage.input_tokens}  output={usage.output_tokens}  "
        f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}  "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}  "
        f"≈ ${cost_of_usage(usage, model):.5f}"
    )
