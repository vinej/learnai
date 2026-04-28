"""
Exercise 3 — Prophet on BTC with a halving regressor.

Bitcoin halvings: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19.
The next is expected around 2028.

Build a Prophet model on log(BTC) daily price with:
- yearly seasonality OFF (it's noise on 24/7 markets at this scale)
- a custom regressor `years_since_halving` (continuous)
- a binary regressor `post_halving_180d` (True for 180 days after each halving)

Compare AIC / log-likelihood and 90-day forecast MAE vs a Prophet
without the regressors. Discuss whether the regressors are justified.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
