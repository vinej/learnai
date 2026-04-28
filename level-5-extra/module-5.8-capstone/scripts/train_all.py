"""Train and pickle all models for one ticker so the API serves quickly.

Usage:
    python scripts/train_all.py --ticker SPY
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from forecaster.config import CACHE
from forecaster.data.loaders import fetch_market
from forecaster.models.ets import ETSForecaster
from forecaster.models.lightgbm_models import LightGBMForecaster
from forecaster.models.naive import NaiveForecaster


def main(ticker: str) -> None:
    out = CACHE / "models" / ticker
    out.mkdir(parents=True, exist_ok=True)

    px = fetch_market(ticker)["Close"].asfreq("B").ffill()
    logret = np.log(px / px.shift(1)).dropna()

    for name, factory in (
        ("naive", lambda: NaiveForecaster()),
        ("ets",   lambda: ETSForecaster(trend=None, seasonal=None)),
        ("lgb",   lambda: LightGBMForecaster(horizon=21)),
    ):
        m = factory()
        m.fit(logret)
        path = out / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(m, f)
        print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    args = p.parse_args()
    main(args.ticker)
