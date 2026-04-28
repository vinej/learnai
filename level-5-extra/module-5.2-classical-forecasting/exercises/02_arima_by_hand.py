"""
Exercise 2 — Identify ARIMA(p,d,q) by hand.

For SPY MONTHLY log returns:
1. Confirm d=0 via ADF.
2. Plot ACF and PACF up to lag 24.
3. Pick (p,q) by inspection. Justify in a comment.
4. Fit your manual ARIMA, then compare to AutoARIMA.
5. Compare AICs and 12-month forecasts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: produce ACF/PACF plots, write your reasoning in a docstring,
# fit manual ARIMA, fit AutoARIMA, compare. Note: monthly equity returns
# usually look like white noise — that is itself a finding.
