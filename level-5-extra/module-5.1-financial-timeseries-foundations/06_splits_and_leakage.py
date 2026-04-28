"""
06 — Time-series splits and look-ahead leakage

Random k-fold CV silently leaks future information into training.
For time series we need:
- A single chronological holdout, OR
- Walk-forward (expanding or rolling) splits.

Common look-ahead leaks:
- Computing standardization stats on the FULL series, then splitting.
- Computing rolling features WITHOUT shifting (current row sees its own value).
- Filling missing values with the WHOLE column's mean.
- Using `shuffle=True` on a time-indexed split.
- Joining macro data on its release date — use the release LAG, not the
  reference date. CPI for March 2024 is published mid-April 2024.

Run: python 06_splits_and_leakage.py
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from _common import fetch_market


def chronological_split(s: pd.Series, train_frac: float = 0.7, val_frac: float = 0.15
                        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(s)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return s.iloc[:n_train], s.iloc[n_train:n_train + n_val], s.iloc[n_train + n_val:]


def walk_forward_splits(s: pd.Series, *, initial_train: int, horizon: int, step: int
                         ) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, test_idx) tuples with EXPANDING training window."""
    n = len(s)
    end = initial_train
    while end + horizon <= n:
        train_idx = s.index[:end]
        test_idx = s.index[end:end + horizon]
        yield train_idx, test_idx
        end += step


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Causal: at time t, use ONLY observations up to t-1."""
    mu = s.shift(1).rolling(window).mean()
    sd = s.shift(1).rolling(window).std()
    return (s - mu) / sd


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    r = np.log(px / px.shift(1)).dropna()

    train, val, test = chronological_split(r, 0.7, 0.15)
    print(f"train: {train.index.min().date()} .. {train.index.max().date()} (n={len(train)})")
    print(f"val:   {val.index.min().date()} .. {val.index.max().date()} (n={len(val)})")
    print(f"test:  {test.index.min().date()} .. {test.index.max().date()} (n={len(test)})")

    print("\n--- Walk-forward splits (1-year horizon, 6-month step) ---")
    for i, (tr, te) in enumerate(walk_forward_splits(
        r, initial_train=252 * 5, horizon=252, step=126
    )):
        print(f"  fold {i}: train n={len(tr)}  test {te.min().date()} .. {te.max().date()}")
        if i >= 3:
            print("  ...")
            break

    # Demonstration of leakage
    print("\n--- Leakage demonstration ---")
    bad_z = (r - r.mean()) / r.std()                    # uses whole-series stats
    good_z = rolling_zscore(r, window=252)              # causal
    print("  bad   (uses future): r[1000] ->", float(bad_z.iloc[1000]))
    print("  good  (rolling):     r[1000] ->", float(good_z.iloc[1000]))
    # The two diverge most when the series' regime in the future is very
    # different from the past — exactly the cases where a model that
    # leaked the global mean will look "great" in backtest and fail live.
