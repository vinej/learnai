"""
Exercise 2 — Three ways to get an 80% PI.

For SPY 21-day-ahead price, compare the 80% prediction interval from:
A) LGB quantile (5.3) at q=0.1 / q=0.9
B) Chronos (5.5) sample-based quantiles
C) Conformal-on-LGB-point (5.7 file 03) at alpha=0.2

Report:
- Empirical coverage on holdout.
- Average interval width.
- Behavior in the high-VIX vs low-VIX subsets.

Conclude which gives the best balance of coverage and tightness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
