"""
03 — Stationarity

A series is (weakly) stationary if its mean, variance, and autocovariance
are time-invariant. ARIMA, regression on past values, and most ML models
assume stationarity (or near-stationarity) of the target.

Two complementary tests:
- Augmented Dickey-Fuller (ADF) — H0: unit root (non-stationary).
  Reject H0 (small p) -> stationary.
- KPSS — H0: stationary (trend or level).
  Reject H0 (small p) -> non-stationary.

Both agree on returns and disagree on prices. That's the lesson.

Run: python 03_stationarity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from _common import fetch_market


def adf_summary(s: pd.Series, label: str) -> None:
    s = s.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    verdict = "stationary" if pval < 0.05 else "NON-stationary"
    print(f"  ADF  {label:<18} stat={stat:+.3f}  p={pval:.4f}  -> {verdict}")


def kpss_summary(s: pd.Series, label: str) -> None:
    s = s.dropna()
    # statsmodels emits warnings about p-value bounds; that's expected.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, *_ = kpss(s, regression="c", nlags="auto")
    verdict = "stationary" if pval > 0.05 else "NON-stationary"
    print(f"  KPSS {label:<18} stat={stat:+.3f}  p={pval:.4f}  -> {verdict}")


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    log_px = np.log(px)
    log_ret = log_px.diff()

    print("--- ADF (H0: unit root, want p<0.05) ---")
    adf_summary(px, "price")
    adf_summary(log_px, "log price")
    adf_summary(log_ret, "log return")

    print("\n--- KPSS (H0: stationary, want p>0.05) ---")
    kpss_summary(px, "price")
    kpss_summary(log_px, "log price")
    kpss_summary(log_ret, "log return")

    # Both tests should agree on log returns: stationary.
    # Both should agree on price: not stationary.
    # If they disagree, you may have trend-stationary data needing
    # detrending rather than differencing — see Hyndman ch. 9.
