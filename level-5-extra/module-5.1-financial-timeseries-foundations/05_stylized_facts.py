"""
05 — Stylized facts of financial returns

Things that are empirically true across most equity, FX, and crypto
return series. Knowing them keeps you honest:

1. Returns are nearly uncorrelated, but |returns| is positively
   autocorrelated (volatility clustering).
2. Returns are leptokurtic — fat tails, much more than a normal.
3. Returns are slightly negatively skewed (equities; crypto less so).
4. Volatility is asymmetric — falls cause bigger vol spikes than
   equivalent rallies (the "leverage effect").
5. Volatility mean-reverts but slowly.
6. Cross-sectional correlations rise during stress.

Run: python 05_stylized_facts.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _common import fetch_market


def stylized_summary(returns: pd.Series, name: str) -> None:
    r = returns.dropna()
    mean = r.mean()
    sd = r.std(ddof=1)
    sk = stats.skew(r)
    kt = stats.kurtosis(r)              # excess kurtosis; 0 = normal
    jb = stats.jarque_bera(r)           # H0: normal

    # Volatility clustering: corr( |r_t|, |r_{t-1}| )
    vol_ac = r.abs().autocorr(lag=1)
    # Return autocorr (should be near zero)
    ret_ac = r.autocorr(lag=1)

    print(f"\n=== {name} ({len(r)} obs) ===")
    print(f"  mean              {mean:+.5f}")
    print(f"  std               {sd:.5f}")
    print(f"  skew              {sk:+.3f}     (negative = left tail heavier)")
    print(f"  excess kurtosis   {kt:+.3f}     (>0 = fatter than normal)")
    print(f"  Jarque-Bera p     {jb.pvalue:.2e}  ({'reject normality' if jb.pvalue < 0.05 else 'cannot reject'})")
    print(f"  AC(1) returns     {ret_ac:+.4f}")
    print(f"  AC(1) |returns|   {vol_ac:+.4f}   (>0 -> vol clusters)")


if __name__ == "__main__":
    for ticker in ("SPY", "BTC-USD", "EURUSD=X"):
        px = fetch_market(ticker)["Close"]
        r = np.log(px / px.shift(1))
        stylized_summary(r, ticker)

    # Takeaways:
    # - Returns aren't normal. Don't size positions assuming they are.
    # - You can probably forecast |return| or variance better than return.
    # - Crypto has fatter tails and higher mean than SPY, with similar AC pattern.
