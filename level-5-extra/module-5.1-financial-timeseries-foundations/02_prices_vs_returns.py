"""
02 — Prices vs returns

We almost always model RETURNS, not prices. Prices are non-stationary
(they trend), have ever-increasing scale, and cross-asset comparison is
meaningless. Returns are scale-free, near-stationary, and additive in
log space — which makes them tractable for stats and ML alike.

Run: python 02_prices_vs_returns.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_market


def simple_returns(prices: pd.Series) -> pd.Series:
    """r_t = P_t / P_{t-1} - 1.  Multiplies across periods."""
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """l_t = ln(P_t / P_{t-1}).  Adds across periods — convenient."""
    return np.log(prices / prices.shift(1))


def annualize_return(daily_log_ret: pd.Series, periods: int = 252) -> float:
    return float(np.exp(daily_log_ret.mean() * periods) - 1)


def annualize_vol(daily_log_ret: pd.Series, periods: int = 252) -> float:
    return float(daily_log_ret.std(ddof=1) * np.sqrt(periods))


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"]
    r = simple_returns(spy).dropna()
    lr = log_returns(spy).dropna()

    print(f"Daily simple return mean: {r.mean():+.6f}  std: {r.std():.6f}")
    print(f"Daily log return    mean: {lr.mean():+.6f}  std: {lr.std():.6f}")

    # Cumulative price = exp(sum of log-returns) — log-returns are additive.
    print(f"\nAnnualized return (geom): {annualize_return(lr):.2%}")
    print(f"Annualized volatility:    {annualize_vol(lr):.2%}")

    reconstructed = np.exp(lr.cumsum()) * float(spy.iloc[0])
    err = float((reconstructed - spy.loc[reconstructed.index]).abs().max())
    print(f"\nMax reconstruction error using log-return sums: {err:.6f}")

    # Common mistake: subtracting prices instead of computing returns.
    # A $1 move in BTC is 1bp; in a $5 stock it's 20%. Always normalize first.
