"""Data loaders — equities/ETF/FX/crypto via yfinance, macro via FRED.

Mirrors module 5.1 _common.py but inside the package.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from fredapi import Fred

from forecaster.config import CACHE, SETTINGS


def fetch_market(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    safe = ticker.replace("=", "_").replace("-", "_").replace("/", "_")
    path = CACHE / f"{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(path)
    return df


def fetch_fred(series_id: str) -> pd.Series:
    path = CACHE / f"fred_{series_id}.parquet"
    if path.exists():
        return pd.read_parquet(path).iloc[:, 0]
    if not SETTINGS.fred_key:
        raise RuntimeError("FRED_API_KEY missing")
    s = Fred(api_key=SETTINGS.fred_key).get_series(series_id)
    s.name = series_id
    s.to_frame().to_parquet(path)
    return s
