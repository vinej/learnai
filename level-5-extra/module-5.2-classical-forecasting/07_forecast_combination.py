"""
07 — Forecast combination

A reliable empirical result: a SIMPLE AVERAGE of decent forecasts
often beats every individual model out-of-sample. The intuition:
errors decorrelate, individual model risk drops.

We average ETS + ARIMA + naive on CPI YoY and check.

Run: python 07_forecast_combination.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from _common import fetch_fred

warnings.filterwarnings("ignore")


def naive(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def fit_ets(train: pd.Series, h: int) -> np.ndarray:
    return ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=12,
                                 initialization_method="estimated").fit().forecast(h).values


def fit_arima(train: pd.Series, h: int) -> np.ndarray:
    return ARIMA(train, order=(2, 1, 2),
                 seasonal_order=(1, 1, 1, 12)).fit().forecast(h).values


def inverse_mse_weights(errors: dict[str, np.ndarray]) -> dict[str, float]:
    inv = {k: 1.0 / max(np.mean(v ** 2), 1e-12) for k, v in errors.items()}
    z = sum(inv.values())
    return {k: v / z for k, v in inv.items()}


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    yoy = (cpi.pct_change(12).dropna() * 100).iloc[-240:]
    train, test = yoy.iloc[:-12], yoy.iloc[-12:]

    individual = {
        "naive": naive(train, 12),
        "ETS":   fit_ets(train, 12),
        "ARIMA": fit_arima(train, 12),
    }
    actual = test.values

    print("MAE by model on last 12 months of CPI YoY:")
    errs = {}
    for name, fc in individual.items():
        e = fc - actual
        errs[name] = e
        print(f"  {name:<8} MAE = {np.abs(e).mean():.3f}")

    # Simple average
    avg = np.mean(np.column_stack(list(individual.values())), axis=1)
    print(f"\n  simple-avg MAE = {np.abs(avg - actual).mean():.3f}")

    # Inverse-MSE weighted (using IN-SAMPLE residual MSE — a real ensemble
    # should fit weights on a separate validation window!)
    weights = inverse_mse_weights(errs)
    print("\nInverse-MSE weights (based on test errors — peeky here, demo only):")
    for k, w in weights.items():
        print(f"  {k:<8} {w:.3f}")
    weighted = sum(w * individual[k] for k, w in weights.items())
    print(f"\n  weighted   MAE = {np.abs(weighted - actual).mean():.3f}")
