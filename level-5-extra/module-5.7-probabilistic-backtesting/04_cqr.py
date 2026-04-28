"""
04 — Conformalized Quantile Regression (CQR).

CQR (Romano et al., 2019) takes a quantile regressor (which gives
ADAPTIVE intervals — wider where the model is less confident) and
adds conformal calibration on top — fixing the coverage guarantee.

Recipe (target alpha = 0.1):
1. Train quantile regressors at q_lo = 0.05 and q_hi = 0.95.
2. On calibration set, compute non-conformity score:
       s_i = max(q_lo_hat_i - y_i, y_i - q_hi_hat_i)
3. q = (1 - alpha)-th adjusted quantile of s_i.
4. Test interval: [q_lo_hat - q, q_hi_hat + q].

This gives the BEST of both worlds: adaptive widths AND coverage.

Run: python 04_cqr.py
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


def fit_q(X, y, q):
    return lgb.train(
        {"objective": "quantile", "alpha": q, "metric": "quantile",
         "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=400,
    )


def cqr(model_lo, model_hi, X_cal, y_cal, X_test, alpha: float = 0.1):
    lo_cal = model_lo.predict(X_cal)
    hi_cal = model_hi.predict(X_cal)
    s = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(s)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q = float(np.quantile(s, q_level))
    return model_lo.predict(X_test) - q, model_hi.predict(X_test) + q


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    n = len(df)
    train = df.iloc[:int(n * 0.6)]
    cal = df.iloc[int(n * 0.6):int(n * 0.8)]
    test = df.iloc[int(n * 0.8):]

    m_lo = fit_q(train[feats], train["y"], 0.05)
    m_hi = fit_q(train[feats], train["y"], 0.95)

    # WITHOUT conformal (raw quantile)
    raw_lo = m_lo.predict(test[feats])
    raw_hi = m_hi.predict(test[feats])
    raw_cov = ((test["y"].values >= raw_lo) & (test["y"].values <= raw_hi)).mean()
    raw_width = float(np.mean(raw_hi - raw_lo))

    # WITH conformal (CQR)
    cqr_lo, cqr_hi = cqr(m_lo, m_hi, cal[feats], cal["y"].values, test[feats], alpha=0.1)
    cqr_cov = ((test["y"].values >= cqr_lo) & (test["y"].values <= cqr_hi)).mean()
    cqr_width = float(np.mean(cqr_hi - cqr_lo))

    print(f"Raw quantile : coverage={raw_cov:.3f}   width={raw_width:.5f}")
    print(f"CQR          : coverage={cqr_cov:.3f}   width={cqr_width:.5f}    (target=0.90)")
    print("\nCQR brings coverage to nominal at the cost of slightly wider intervals.")
