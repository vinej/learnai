"""
Exercise 2 — Stationarity report.

For each of {SPY price, SPY log price, SPY log return,
EUR/USD log return, BTC log return, DGS10 (10Y yield), DGS10 first diff}:
- Run ADF and KPSS.
- Print a small markdown table with verdicts.
- Conclude in 3-4 sentences:
    Why is DGS10 borderline?
    Why is BTC log return stationary despite having ridiculous tails?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: Reuse adf_summary / kpss_summary from 03_stationarity.py
# (copy them in or import via importlib.util — module names with leading
# digits aren't importable normally).
