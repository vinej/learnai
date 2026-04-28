"""
01 — Baselines

Four baselines worth keeping for life:
- Naive    : y_hat_{t+h} = y_t
- Drift    : naive plus the average per-period change
- SeasonalNaive : y_hat_{t+h} = y_{t+h-m}  for season length m
- Mean     : y_hat = mean(train)

If a deep learning model can't beat ALL of these, it's not learning.

Run: python 01_baselines.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_fred, fetch_market


def naive_forecast(train: pd.Series, h: int) -> pd.Series:
    return pd.Series([float(train.iloc[-1])] * h)


def drift_forecast(train: pd.Series, h: int) -> pd.Series:
    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / max(1, len(train) - 1)
    last = float(train.iloc[-1])
    return pd.Series([last + slope * (i + 1) for i in range(h)])


def seasonal_naive_forecast(train: pd.Series, h: int, m: int) -> pd.Series:
    last_season = train.iloc[-m:].values
    return pd.Series([float(last_season[i % m]) for i in range(h)])


def mean_forecast(train: pd.Series, h: int) -> pd.Series:
    return pd.Series([float(train.mean())] * h)


def mae(yhat: pd.Series, y: pd.Series) -> float:
    return float(np.abs(yhat.values - y.values).mean())


def mape(yhat: pd.Series, y: pd.Series) -> float:
    return float(np.abs((yhat.values - y.values) / y.values).mean())


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()  # monthly, start
    yoy = cpi.pct_change(12).dropna() * 100

    train, test = yoy.iloc[:-12], yoy.iloc[-12:]
    h = len(test)

    forecasts = {
        "naive": naive_forecast(train, h),
        "drift": drift_forecast(train, h),
        "seasonal_naive(12)": seasonal_naive_forecast(train, h, m=12),
        "mean": mean_forecast(train, h),
    }
    print("CPI YoY 12-month forecasts (last year of data as test):\n")
    for name, fc in forecasts.items():
        fc.index = test.index
        print(f"  {name:<22} MAE={mae(fc, test):.3f}  MAPE={mape(fc, test):.2%}")

    # On a slow-moving series like inflation, naive is hard to beat.
    # On equities, naive on returns -> 0 return forecast. Hard to beat there too.
