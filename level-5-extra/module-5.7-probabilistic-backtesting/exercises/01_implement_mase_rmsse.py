"""
Exercise 1 — Implement MASE and RMSSE.

Implement them from scratch, then verify against:
- sktime.performance_metrics.forecasting (if installed), or
- a manual calculation on a small synthetic example.

Compute both on (a) SPY 1-day return forecasts from 5.3 LGB, (b) CPI
YoY 12-month forecasts from 5.2 ETS. Discuss what the values mean
relative to the in-sample naive baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and verify.
