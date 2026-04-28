"""
Exercise 2 — Sample-based quantiles from Chronos vs LGB quantile.

1. Sample 500 paths from Chronos for SPY 21d-ahead price.
2. Compute p10/p50/p90 from samples.
3. Train LGB quantile (q in 0.1, 0.5, 0.9) on the same 21d-ahead target.
4. Compare:
   - MAE on p50.
   - 80% empirical coverage on the same holdout.
   - Average interval width.

Which is better calibrated? Which has tighter intervals?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
