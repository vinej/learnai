"""Forecast and trading metrics — central definitions used across the package."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, yhat): return float(np.mean(np.abs(np.asarray(yhat) - np.asarray(y))))
def rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(yhat) - np.asarray(y)) ** 2)))


def mase(y, yhat, y_train, m: int = 1) -> float:
    naive = float(np.mean(np.abs(np.diff(np.asarray(y_train), n=m))))
    return mae(y, yhat) / max(naive, 1e-12)


def coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def crps(samples: np.ndarray, y_true: float) -> float:
    s = np.asarray(samples)
    return float(np.mean(np.abs(s - y_true)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])))


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(periods)) if sd else 0.0


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())
