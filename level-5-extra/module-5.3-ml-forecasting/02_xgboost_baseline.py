"""
02 — XGBoost baseline for next-day return.

Use the feature matrix from 01_feature_engineering.py. Train an XGB
regressor on a 70/15/15 chronological split. Report MAE, RMSE,
direction accuracy, and check vs the zero-prediction baseline.

Lesson you'll see: a good XGB model on raw features will still struggle
to beat a zero forecast on RMSE, because daily returns are mostly noise.
What it CAN do is shift the conditional mean a tiny amount AND give you
a usable signal that's worth combining with others.

Run: python 02_xgboost_baseline.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from _common import fetch_market

# Reuse the feature builder by importing via importlib trickery — module
# names with leading digits aren't directly importable.
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "fe", pathlib.Path(__file__).parent / "01_feature_engineering.py"
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


def chrono_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)

    feature_cols = [c for c in df.columns if c not in ("close", "y")]
    train, val, test = chrono_split(df)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=30,
    )
    model.fit(train[feature_cols], train["y"],
              eval_set=[(val[feature_cols], val["y"])],
              verbose=False)

    pred = model.predict(test[feature_cols])
    actual = test["y"].values

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    dacc = float(np.mean(np.sign(pred) == np.sign(actual)))

    # Zero baseline
    zero_mae = float(np.abs(actual).mean())
    zero_rmse = float(np.sqrt(np.mean(actual ** 2)))

    print(f"XGB    MAE={mae:.5f}  RMSE={rmse:.5f}  DirAcc={dacc:.3f}")
    print(f"Zero   MAE={zero_mae:.5f}  RMSE={zero_rmse:.5f}  DirAcc=0.500")
    print(f"\nDelta vs zero:  MAE {mae - zero_mae:+.6f}   RMSE {rmse - zero_rmse:+.6f}")
    print(f"Trees used (best_iteration): {model.best_iteration}")

    # Top 10 features by gain
    booster = model.get_booster()
    imp = booster.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 features by gain:")
    for name, score in top:
        print(f"  {name:<25} {score:.2f}")
