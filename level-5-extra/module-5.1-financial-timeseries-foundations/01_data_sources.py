"""
01 — Data sources

Pull the four asset classes we use throughout Level 5:
- Equities/ETFs (yfinance)
- FX (yfinance, "EURUSD=X")
- Crypto (yfinance, "BTC-USD")
- Macro (FRED: CPI, unemployment, fed funds, 10Y yield)

The actual loaders live in `_common.py` so other files (and other
sub-modules) can import them. Module names starting with a digit are
not importable in Python — this is why we keep helpers in `_common.py`.

Run: python 01_data_sources.py
"""
from __future__ import annotations

from _common import CACHE, fetch_fred, fetch_market

if __name__ == "__main__":
    spy = fetch_market("SPY")
    btc = fetch_market("BTC-USD")
    eur = fetch_market("EURUSD=X")
    print("SPY :", spy.shape, "from", spy.index.min().date(), "to", spy.index.max().date())
    print("BTC :", btc.shape)
    print("EUR :", eur.shape)

    cpi = fetch_fred("CPIAUCSL")          # CPI, monthly
    unrate = fetch_fred("UNRATE")         # Unemployment, monthly
    fedfunds = fetch_fred("DFF")          # Fed funds rate, daily
    dgs10 = fetch_fred("DGS10")           # 10Y Treasury yield, daily
    print("CPI    :", cpi.shape, "latest=", cpi.tail(1).iloc[0])
    print("UNRATE :", unrate.shape)
    print("DFF    :", fedfunds.shape)
    print("DGS10  :", dgs10.shape)

    print("\nCached in", CACHE)
