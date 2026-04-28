"""
06 — Walk-forward backtest scaffold.

Anchored vs rolling:
- ANCHORED: training window grows; first day is fixed.
- ROLLING:  training window slides; size constant.

Both yield a stream of (forecast_t, actual_t) pairs over disjoint
test windows, refit per window. This is what you should report on,
NOT a single fixed train/test split.

Run: python 06_walk_forward_backtest.py
"""
from __future__ import annotations

import importlib.util
import pathlib
from collections.abc import Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd

from _common import fetch_market

_FE = (pathlib.Path(__file__).resolve().parents[1]
        / "module-5.3-ml-forecasting" / "01_feature_engineering.py")
_spec = importlib.util.spec_from_file_location("fe", _FE)
fe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fe)


def walk_forward(df: pd.DataFrame, *, initial_train: int, horizon: int,
                  step: int, mode: str = "anchored") -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    end = initial_train
    while end + horizon <= n:
        if mode == "anchored":
            tr = df.iloc[:end]
        elif mode == "rolling":
            tr = df.iloc[max(0, end - initial_train):end]
        else:
            raise ValueError(mode)
        te = df.iloc[end:end + horizon]
        yield tr, te
        end += step


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    booster = lgb.train(
        {"objective": "regression_l1", "learning_rate": 0.03,
         "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
        lgb.Dataset(train[feats], label=train["y"]),
        num_boost_round=300,
    )
    return booster.predict(test[feats])


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    preds, actuals, fold_dates = [], [], []
    for tr, te in walk_forward(df, initial_train=252 * 5, horizon=63, step=63, mode="anchored"):
        p = fit_predict(tr, te, feats)
        preds.extend(p); actuals.extend(te["y"].values)
        fold_dates.append((te.index.min().date(), te.index.max().date()))

    preds = np.array(preds); actuals = np.array(actuals)
    print(f"Folds run: {len(fold_dates)}")
    print(f"Period: {fold_dates[0][0]} -> {fold_dates[-1][1]}")
    print(f"MAE  = {np.abs(preds - actuals).mean():.5f}")
    print(f"RMSE = {np.sqrt(((preds - actuals) ** 2).mean()):.5f}")
    print(f"DAcc = {(np.sign(preds) == np.sign(actuals)).mean():.3f}")
