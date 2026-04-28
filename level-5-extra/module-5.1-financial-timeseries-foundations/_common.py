"""
_common.py — shared data loaders for Module 5.1.

Other modules in Level 5 import these too, via:
    sys.path.append(str(Path(__file__).parent.parent / "module-5.1-financial-timeseries-foundations"))
    from _common import fetch_market, fetch_fred
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CACHE = Path(__file__).parent / "data_cache"
CACHE.mkdir(exist_ok=True)


def fetch_market(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Daily OHLCV for any yfinance-supported symbol; cached as Parquet."""
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
    """One FRED series; cached as Parquet."""
    path = CACHE / f"fred_{series_id}.parquet"
    if path.exists():
        return pd.read_parquet(path).iloc[:, 0]
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY missing — see Module 5.1 INSTALL.md")
    s = Fred(api_key=api_key).get_series(series_id)
    s.name = series_id
    s.to_frame().to_parquet(path)
    return s
