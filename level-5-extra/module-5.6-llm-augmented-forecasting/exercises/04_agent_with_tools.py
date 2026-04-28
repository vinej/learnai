"""
Exercise 4 — A research agent that calls forecasting tools.

Build a single Claude agent with tools:
  - get_price_history(ticker, period)        -> CSV via yfinance
  - get_macro_series(series_id)               -> CSV via FRED
  - run_lgb_forecast(target, horizon)         -> point + 80% PI
  - run_chronos_forecast(target, horizon)     -> point + 80% PI

User prompt: "Give me a 5-day directional forecast for SPY with
reasoning. Use both LGB and Chronos. Reconcile if they disagree."

Log the full tool-call trace. Print the final structured answer.

Bonus: add a tool `web_search` (or just a stub) so the agent can pull
recent news, and have it cite a source for its directional view.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement using the Anthropic tool_use API (see L4.4).
