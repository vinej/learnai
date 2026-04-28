"""
03 — LightGBM quantile regression for prediction intervals.

Train three LightGBM models with objective="quantile" at q=0.1, 0.5, 0.9.
Together they give a point forecast (median) and an 80% prediction
interval. Conformal calibration in 5.7 will tighten this.

Empirical coverage on holdout should be near 80%. If it's much lower,
the model is overconfident; if much higher, the intervals are loose.

Run: python 03_lightgbm_quantile.py
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


def fit_quantile(train_X, train_y, val_X, val_y, q: float) -> lgb.Booster:
    return lgb.train(
        params={
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "verbose": -1,
        },
        train_set=lgb.Dataset(train_X, label=train_y),
        valid_sets=[lgb.Dataset(val_X, label=val_y)],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    n = len(df)
    train = df.iloc[:int(n * 0.7)]
    val = df.iloc[int(n * 0.7):int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]

    models = {q: fit_quantile(train[feats], train["y"], val[feats], val["y"], q)
              for q in (0.1, 0.5, 0.9)}

    pred = pd.DataFrame({q: m.predict(test[feats]) for q, m in models.items()},
                         index=test.index)
    actual = test["y"].values

    inside = ((actual >= pred[0.1].values) & (actual <= pred[0.9].values)).mean()
    median_mae = float(np.abs(pred[0.5].values - actual).mean())
    width_avg = float((pred[0.9] - pred[0.1]).mean())

    print(f"Empirical 80% coverage: {inside:.3f}     (target = 0.80)")
    print(f"Median forecast MAE   : {median_mae:.5f}")
    print(f"Average interval width: {width_avg:.5f}")

    # If coverage is much below 80%, conformal prediction in 5.7 will fix it.
