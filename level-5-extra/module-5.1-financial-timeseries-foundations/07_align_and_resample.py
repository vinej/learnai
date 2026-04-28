"""
07 — Aligning mixed-frequency data

Reality: prices are daily (with weekends gone for stocks, no gaps for
crypto), CPI is monthly, GDP is quarterly, FX is 24x5. Combining them
correctly without leakage is the tricky bit.

Rules of thumb:
- Forward-fill SLOWER series onto the FASTER index (a CPI release stays
  the official value until the next release).
- Use the RELEASE date, not the reference date. CPI for March is
  published ~mid-April. If you join on March 31, you're peeking.
- Resample with `.resample('W-FRI').last()` etc. — explicit about the
  closing convention.

Run: python 07_align_and_resample.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_fred, fetch_market


def align_macro_to_daily(daily_idx: pd.DatetimeIndex, macro: pd.Series,
                          release_lag_days: int = 30) -> pd.Series:
    """
    Shift macro forward by its publication lag, then forward-fill onto
    a daily index. Conservative — better to be late than to peek.
    """
    shifted = macro.copy()
    shifted.index = shifted.index + pd.Timedelta(days=release_lag_days)
    return shifted.reindex(daily_idx, method="ffill")


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"]
    cpi = fetch_fred("CPIAUCSL")
    dgs10 = fetch_fred("DGS10")

    daily_idx = spy.index

    cpi_daily = align_macro_to_daily(daily_idx, cpi, release_lag_days=30)
    yld_daily = dgs10.reindex(daily_idx, method="ffill")  # daily series, just align

    df = pd.DataFrame({
        "spy": spy,
        "spy_logret": np.log(spy / spy.shift(1)),
        "cpi": cpi_daily,
        "cpi_yoy": cpi_daily.pct_change(252),
        "dgs10": yld_daily,
    }).dropna()

    print(df.tail(5).round(3))
    print("\nshape:", df.shape)
    print("date range:", df.index.min().date(), "->", df.index.max().date())

    # Sanity: cpi values should be flat for stretches (it's monthly, ffilled)
    runs = (df["cpi"].diff().fillna(0) == 0).astype(int).groupby(
        df["cpi"].diff().fillna(0).ne(0).cumsum()
    ).sum()
    print("\nLongest run of unchanged CPI (days):", int(runs.max()))
    print("Mean run length:", float(runs.mean()))
    # Should be ~21 trading days, matching one calendar month.
