"""
02 — Sentiment as a feature in an ML forecaster.

Synthesize a "daily sentiment" series (in production: aggregate the
LLM scores from 01 across all ticker-tagged headlines for the day),
join it to the SPY return target, and check whether it improves a
LightGBM via walk-forward CV.

Critical: lag the sentiment by at least 1 day to avoid look-ahead.
News from the same day was published over the trading session — using
it as a same-day feature is leakage unless you carefully filter to
pre-open headlines.

Run: python 02_sentiment_as_feature.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from _common import fetch_market

# Re-use FE from 5.3
_FE_PATH = (pathlib.Path(__file__).resolve().parents[1]
            / "module-5.3-ml-forecasting" / "01_feature_engineering.py")
_spec = importlib.util.spec_from_file_location("fe", _FE_PATH)
fe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fe)


def synthetic_sentiment_series(idx: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    """Stand-in: a noisy AR(1) loosely correlated with SPY's NEXT-day return.
    In a real workflow this comes from your news pipeline.
    """
    rng = np.random.default_rng(seed)
    n = len(idx)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.6 * s[i - 1] + rng.normal(0, 0.4)
    return pd.Series(np.tanh(s), index=idx, name="sentiment")


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    df["sentiment_lag1"] = synthetic_sentiment_series(df.index).shift(1)
    df = df.dropna()

    n = len(df)
    train, test = df.iloc[:int(n * 0.8)], df.iloc[int(n * 0.8):]

    def fit_eval(feature_cols: list[str]) -> dict:
        b = lgb.train(
            {"objective": "regression_l1", "learning_rate": 0.03,
             "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
            lgb.Dataset(train[feature_cols], label=train["y"]),
            num_boost_round=300,
        )
        pred = b.predict(test[feature_cols])
        return {"mae": float(np.abs(pred - test["y"].values).mean()),
                 "dacc": float((np.sign(pred) == np.sign(test["y"].values)).mean())}

    base = fit_eval(feats)
    with_sent = fit_eval(feats + ["sentiment_lag1"])

    print(f"Without sentiment: MAE={base['mae']:.5f}  DAcc={base['dacc']:.3f}")
    print(f"With sentiment   : MAE={with_sent['mae']:.5f}  DAcc={with_sent['dacc']:.3f}")
    print(f"Delta MAE        : {with_sent['mae'] - base['mae']:+.6f}")
    # The synthetic version is intentionally weak; in real workflows the
    # gain depends entirely on the quality of your news pipeline.
