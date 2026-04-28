"""
Exercise 5 — Reproducible head-to-head: LGB vs LSTM vs N-BEATS vs PatchTST.

Pick ONE target from {SPY 1d return, BTC 5d return, EUR/USD 5d return,
SPY 21d realized vol, CPI YoY 12 months}.

Same data, same chronological split, same loss (MAE). Train each model
with reasonable defaults; record training time, parameter count, MAE,
RMSE.

Output a Markdown table and a 5-sentence conclusion: which won, by
how much, and was the difference statistically meaningful given the
holdout size?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement. This is a multi-hour exercise the first time you do
# it; it's the foundation of the capstone in 5.8.
