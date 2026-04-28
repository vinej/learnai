"""
Exercise 5 — Regime-stratified backtest.

Take the backtest from Exercise 4. Partition the holdout by VIX
quartile (Q1 = lowest vol, Q4 = highest). Recompute Sharpe, MAE,
direction accuracy, max drawdown, and 80% PI coverage WITHIN each
quartile.

Discuss:
- Where does the strategy work / fail?
- Does coverage drop in high-vol periods? (Usually yes — motivates
  adaptive conformal.)
- How would you reduce gross exposure in Q4 to prevent drawdowns?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
