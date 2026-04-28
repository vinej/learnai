"""
05 — Walk-forward CV with sklearn's TimeSeriesSplit.

`TimeSeriesSplit` enforces a chronological train/test split per fold.
With `expanding` window, each fold trains on more data than the last.
Combined with refit-each-fold, you get an honest estimate of the
distribution of out-of-sample errors.

Pitfalls:
- Default `n_splits=5` may be too few for short series.
- Use `gap` to avoid leak when features touch labels at near-boundary.
- Always refit. Reusing a model across folds is leakage.

Run: python 05_walk_forward_cv.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from _common import fetch_market

_spec = importlib.util.spec_from_file_location(
    "fe", pathlib.Path(__file__).parent / "01_feature_engineering.py"
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    tscv = TimeSeriesSplit(n_splits=8, gap=5, test_size=252)
    fold_results = []
    for i, (tr_idx, te_idx) in enumerate(tscv.split(df)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        booster = lgb.train(
            {"objective": "regression_l1", "learning_rate": 0.03,
             "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
            lgb.Dataset(tr[feats], label=tr["y"]),
            num_boost_round=300,
        )
        pred = booster.predict(te[feats])
        actual = te["y"].values
        mae = float(np.abs(pred - actual).mean())
        dacc = float((np.sign(pred) == np.sign(actual)).mean())
        fold_results.append((i, te.index.min().date(), te.index.max().date(), mae, dacc))
        print(f"Fold {i}  test {te.index.min().date()} .. {te.index.max().date()}  "
              f"MAE={mae:.5f}  DirAcc={dacc:.3f}")

    fr = pd.DataFrame(fold_results, columns=["fold", "start", "end", "mae", "dacc"])
    print(f"\nMean MAE  {fr['mae'].mean():.5f}  (sd {fr['mae'].std():.5f})")
    print(f"Mean DAcc {fr['dacc'].mean():.3f}  (sd {fr['dacc'].std():.3f})")
