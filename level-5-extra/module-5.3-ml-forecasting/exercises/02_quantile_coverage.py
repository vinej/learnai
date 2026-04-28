"""
Exercise 2 — Quantile coverage check.

Train LGB quantile regressors at q in {0.05, 0.5, 0.95}.
Evaluate empirical coverage of the 90% PI on:
- The full holdout
- The "high vol" quantile of holdout (top 25% by realized 21-day vol)
- The "low vol" quantile

You will likely see coverage is too LOW in high-vol regimes — the
intervals don't widen enough. This motivates conformal prediction in 5.7.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and write 5-line conclusion about regime-conditional miscoverage.
