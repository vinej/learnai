"""
04 — ACF, PACF, seasonality

ACF(k)  — correlation of x_t with x_{t-k}.
PACF(k) — correlation after removing the linear effect of lags 1..k-1.

For daily equity returns, ACF should be near zero — markets are roughly
unforecastable at the price-return level. But ACF of *squared* or
*absolute* returns is positive: volatility clusters. That asymmetry
drives a lot of financial modeling.

Run: python 04_acf_pacf_seasonality.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from _common import fetch_market


def show_acf(s: pd.Series, label: str, nlags: int = 10) -> None:
    s = s.dropna()
    a = acf(s, nlags=nlags, fft=True)
    formatted = "  ".join(f"{v:+.3f}" for v in a[1:])
    print(f"  ACF  {label:<18} lags 1..{nlags}: {formatted}")


def show_pacf(s: pd.Series, label: str, nlags: int = 10) -> None:
    s = s.dropna()
    p = pacf(s, nlags=nlags, method="ols")
    formatted = "  ".join(f"{v:+.3f}" for v in p[1:])
    print(f"  PACF {label:<18} lags 1..{nlags}: {formatted}")


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"]
    log_ret = np.log(spy / spy.shift(1)).dropna()

    print("--- Daily log returns ---")
    show_acf(log_ret, "returns")
    show_acf(log_ret ** 2, "squared returns")     # volatility clustering
    show_acf(log_ret.abs(), "abs returns")

    print("\n--- PACF for returns vs squared returns ---")
    show_pacf(log_ret, "returns")
    show_pacf(log_ret ** 2, "squared returns")

    # Calendar features (built without leakage — they only depend on the date)
    df = pd.DataFrame({"r": log_ret})
    df["dow"] = df.index.dayofweek                # 0=Mon
    df["month"] = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

    print("\n--- Mean return by day of week ---")
    print((df.groupby("dow")["r"].mean() * 1e4).round(2).rename("bps"))
    # Tiny numbers, often inside the noise. Don't trade on them.
