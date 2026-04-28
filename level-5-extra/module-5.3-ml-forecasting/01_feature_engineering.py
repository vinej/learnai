"""
01 — Feature engineering for time-series ML.

Five families of features. All MUST be computed from values strictly
before time t (use .shift(1) before any rolling op).

1. Lags                — y_{t-1}, y_{t-2}, ...
2. Rolling stats       — mean / std / min / max / quantile of y over k periods
3. Differences         — y_t - y_{t-k}
4. Calendar            — dayofweek, month, is_month_end, is_quarter_end
5. Exogenous / regime  — macro, sector relative, vol regime, distance to all-time high

Run: python 01_feature_engineering.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_market


def add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    for k in lags:
        df[f"{col}_lag{k}"] = df[col].shift(k)
    return df


def add_rolling(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    base = df[col].shift(1)         # CRITICAL: shift before rolling
    for w in windows:
        df[f"{col}_rmean{w}"] = base.rolling(w).mean()
        df[f"{col}_rstd{w}"] = base.rolling(w).std()
        df[f"{col}_rmin{w}"] = base.rolling(w).min()
        df[f"{col}_rmax{w}"] = base.rolling(w).max()
    return df


def add_diffs(df: pd.DataFrame, col: str, ks: list[int]) -> pd.DataFrame:
    for k in ks:
        df[f"{col}_diff{k}"] = df[col].diff(k)
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    df["is_year_end"] = idx.is_year_end.astype(int)
    return df


def add_distance_to_high(df: pd.DataFrame, price_col: str, lookback: int = 252) -> pd.DataFrame:
    rolling_high = df[price_col].shift(1).rolling(lookback).max()
    df[f"dist_to_high_{lookback}"] = df[price_col] / rolling_high - 1
    return df


def build_feature_matrix(px: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": px})
    df["logret"] = np.log(df["close"] / df["close"].shift(1))

    add_lags(df, "logret", [1, 2, 3, 5, 10, 21])
    add_rolling(df, "logret", [5, 10, 21, 63])
    add_rolling(df, "close", [21, 63])
    add_diffs(df, "close", [1, 5, 21])
    add_calendar(df)
    add_distance_to_high(df, "close", 252)

    # Target: next-day log return
    df["y"] = df["logret"].shift(-1)
    return df.dropna()


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"]
    feats = build_feature_matrix(spy)
    print("Feature matrix shape:", feats.shape)
    print("Columns:", list(feats.columns)[:10], "...")
    print("\nHead:")
    print(feats.tail(3).round(4))
