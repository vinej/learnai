"""
02 — Exponential smoothing (ETS)

Three flavors:
- Simple ES:  level only.
- Holt:       level + trend.
- Holt-Winters: level + trend + seasonality (additive or multiplicative).

ETS is one of the most reliably good baselines on real business data
in the M-competitions. State-space form gives prediction intervals.

Run: python 02_exponential_smoothing.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from _common import fetch_fred

warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    yoy = (cpi.pct_change(12).dropna() * 100).iloc[-240:]   # last 20Y

    train, test = yoy.iloc[:-12], yoy.iloc[-12:]

    # Holt-Winters with additive trend, additive seasonality, m=12
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()

    fc = model.forecast(12)
    fc.index = test.index

    err = test - fc
    print("Holt-Winters add+add forecast for last 12 months of CPI YoY:\n")
    out = pd.DataFrame({"actual": test, "forecast": fc, "error": err}).round(3)
    print(out)
    print(f"\nMAE: {err.abs().mean():.3f}")
    print(f"AIC: {model.aic:.1f}")

    # Try Holt (no seasonality) too
    holt = ExponentialSmoothing(train, trend="add", seasonal=None,
                                 initialization_method="estimated").fit()
    fc2 = holt.forecast(12)
    fc2.index = test.index
    print(f"Holt (no seasonality) MAE: {(test - fc2).abs().mean():.3f}")

    # Lesson: for CPI YoY, seasonal component helps a little but isn't huge.
    # For monthly retail sales it would dominate.
