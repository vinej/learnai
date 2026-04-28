"""
08 — Trading metrics.

Sharpe, Sortino, max drawdown, hit rate, expectancy, Calmar.
We also show how `quantstats` collapses all of this into a one-liner
report.

Run: python 08_trading_metrics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_market


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods))


def sortino(returns: pd.Series, periods: int = 252) -> float:
    downside = returns[returns < 0].std(ddof=1)
    if downside == 0 or np.isnan(downside):
        return float("inf")
    return float(returns.mean() / downside * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def calmar(returns: pd.Series, periods: int = 252) -> float:
    ann = (1 + returns.mean()) ** periods - 1
    mdd = abs(max_drawdown(returns))
    return float(ann / mdd) if mdd > 0 else float("inf")


def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def expectancy(returns: pd.Series) -> float:
    win = returns[returns > 0].mean()
    loss = returns[returns < 0].mean()
    p_win = (returns > 0).mean()
    if pd.isna(loss):
        return float(p_win * win)
    return float(p_win * win + (1 - p_win) * loss)


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"]
    r = spy.pct_change().dropna()
    print(f"SPY buy-and-hold")
    print(f"  Sharpe   : {sharpe(r):.2f}")
    print(f"  Sortino  : {sortino(r):.2f}")
    print(f"  MaxDD    : {max_drawdown(r):.2%}")
    print(f"  Calmar   : {calmar(r):.2f}")
    print(f"  HitRate  : {hit_rate(r):.2%}")
    print(f"  Expect   : {expectancy(r):+.5f} per day")

    # Quantstats one-liner — saves an HTML tearsheet
    try:
        import quantstats as qs
        qs.extend_pandas()
        # qs.reports.html(r, output="spy_tearsheet.html", title="SPY")
        print("\nQuickstats: ", qs.stats.sharpe(r), "Sharpe;",
              f"{qs.stats.max_drawdown(r):.2%}", "MDD")
    except ImportError:
        pass
