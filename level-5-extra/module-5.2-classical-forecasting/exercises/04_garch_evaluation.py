"""
Exercise 4 — GARCH evaluation.

Fit GARCH(1,1) and EGARCH(1,1) on SPY daily log returns.
Walk-forward 1-day-ahead variance forecasts over the last 500 days.
Evaluate using:
- MSE(sigma2_hat, r^2)
- QLIKE loss = sigma2_hat / r^2 - log(sigma2_hat / r^2) - 1
  (Patton 2011 — robust to noise in the volatility proxy)
- correlation between sigma_hat and |r|

Bonus: do the same for BTC and compare. Crypto vol is rarely
well-described by GARCH alone — note where it fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and write 5-line conclusion.
