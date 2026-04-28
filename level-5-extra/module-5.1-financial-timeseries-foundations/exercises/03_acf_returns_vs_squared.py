"""
Exercise 3 — ACF: returns vs squared returns.

Goal:
- Compute ACF (lags 1..40) for SPY daily log returns AND for squared
  returns.
- Plot both on the same matplotlib figure with the 95% CI bands
  (~ ±1.96/sqrt(N)).
- In a docstring, explain why one looks like white noise and the other
  is clearly autocorrelated. What does that imply about which is easier
  to forecast?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: Use statsmodels.graphics.tsaplots.plot_acf or compute ACF manually.
# TODO: Save figure to acf_returns_vs_squared.png in this folder.
