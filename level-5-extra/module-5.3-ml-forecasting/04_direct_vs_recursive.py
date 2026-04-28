"""
04 — Direct vs recursive multi-step forecasting.

Two strategies for h-step-ahead forecasting:

RECURSIVE
  Train one 1-step model. To predict t+h, feed predictions back as
  features. Errors compound, but you only train one model.

DIRECT
  Train h separate models, one per horizon. y_target_k = y_{t+k}.
  No error compounding, but more models and the long-horizon ones
  see less effective signal.

Empirically: direct usually wins on noisy series at horizons up to ~20.
Recursive shines when you have a great 1-step model and short horizons.

Run: python 04_direct_vs_recursive.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from _common import fetch_market

_spec = importlib.util.spec_from_file_location(
    "fe", pathlib.Path(__file__).parent / "01_feature_engineering.py"
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


HORIZONS = [1, 2, 5, 10, 21]


def fit_lgb(X, y) -> lgb.Booster:
    return lgb.train(
        {"objective": "regression_l1", "learning_rate": 0.03,
         "num_leaves": 31, "min_data_in_leaf": 50, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=300,
    )


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px).drop(columns=["y"])  # rebuild target ourselves
    feats = [c for c in df.columns if c not in ("close",)]

    # Targets: cumulative log return over h days
    logret = np.log(df["close"] / df["close"].shift(1))
    for h in HORIZONS:
        df[f"y_h{h}"] = logret.rolling(h).sum().shift(-h)

    df = df.dropna()
    n = len(df)
    train = df.iloc[:int(n * 0.8)]
    test = df.iloc[int(n * 0.8):]

    print("=== DIRECT (one model per horizon) ===")
    direct_mae = {}
    for h in HORIZONS:
        m = fit_lgb(train[feats], train[f"y_h{h}"])
        pred = m.predict(test[feats])
        mae = float(np.abs(pred - test[f"y_h{h}"].values).mean())
        direct_mae[h] = mae
        print(f"  h={h:>2}  MAE={mae:.5f}")

    print("\n=== RECURSIVE (one 1-step model, fed back) ===")
    m1 = fit_lgb(train[feats], train["y_h1"])
    # Recursive simulation only makes sense if our features can be updated
    # with synthetic predictions. Here we approximate by computing the
    # cumulative pred over h steps using h consecutive 1-step preds on
    # the test set's actual lagged features (a slight cheat — see
    # exercise 3 for an honest recursive feature roll).
    p1 = m1.predict(test[feats])
    for h in HORIZONS:
        # naive recursive: assume the same drift each day for h days
        pred_h = p1 * h
        mae = float(np.abs(pred_h - test[f"y_h{h}"].values).mean())
        print(f"  h={h:>2}  MAE={mae:.5f}   (DIRECT was {direct_mae[h]:.5f})")

    # On equity returns the gap is small at h=1 and grows with h.
