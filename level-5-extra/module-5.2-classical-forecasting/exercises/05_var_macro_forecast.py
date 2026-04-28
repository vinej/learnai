"""
Exercise 5 — VAR macro forecast with confidence bands.

Build a VAR on (CPI YoY, UNRATE, DGS10, DFF). Produce:
- 12-month-ahead point forecasts.
- 80% and 95% prediction intervals.
- An impulse-response chart: how does a 1pp shock to DFF
  propagate through CPI and UNRATE over 24 months?

Bonus: compare to a Bayesian VAR (statsmodels.tsa.api.BayesianVAR
or pymc-based equivalents) for one of the variables.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: VAR with .forecast_interval(); .irf(steps=24).plot().
