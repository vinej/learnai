"""
03 — Split conformal prediction.

Distribution-free prediction intervals with finite-sample coverage
guarantees, given exchangeability.

Recipe (regression, target alpha = 0.1 -> 90% PI):
1. Train ANY point-forecast model on a training set.
2. On a held-out CALIBRATION set, compute residuals s_i = |y_i - y_hat_i|.
3. Take q = the (n_cal+1)*(1-alpha)/n_cal-th quantile of residuals.
4. New prediction: [y_hat - q, y_hat + q].

Coverage is guaranteed >= 1-alpha marginally over the test distribution
provided exchangeability — financial data is approximately exchangeable
within a regime, often broken by regime shifts. Hence "adaptive
conformal" (file 05).

Run: python 03_split_conformal.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np

from _common import fetch_market

_FE = (pathlib.Path(__file__).resolve().parents[1]
        / "module-5.3-ml-forecasting" / "01_feature_engineering.py")
_spec = importlib.util.spec_from_file_location("fe", _FE)
fe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fe)


def split_conformal(model, X_cal, y_cal, X_test, alpha: float = 0.1):
    pred_cal = model.predict(X_cal)
    residuals = np.abs(y_cal - pred_cal)
    n = len(residuals)
    # The "+1" correction so coverage >= 1 - alpha exactly under exchangeability
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q = float(np.quantile(residuals, q_level))
    pred_test = model.predict(X_test)
    return pred_test - q, pred_test, pred_test + q


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    n = len(df)
    train = df.iloc[:int(n * 0.6)]
    cal = df.iloc[int(n * 0.6):int(n * 0.8)]
    test = df.iloc[int(n * 0.8):]

    booster = lgb.train(
        {"objective": "regression_l2", "learning_rate": 0.03,
         "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
        lgb.Dataset(train[feats], label=train["y"]),
        num_boost_round=300,
    )
    lo, mid, hi = split_conformal(booster, cal[feats], cal["y"].values, test[feats], alpha=0.1)

    inside = ((test["y"].values >= lo) & (test["y"].values <= hi)).mean()
    width = float(np.mean(hi - lo))
    print(f"Empirical 90% coverage: {inside:.3f}   (target = 0.90)")
    print(f"Average interval width: {width:.5f}")
    # Coverage should be very close to 0.90 under exchangeability.
