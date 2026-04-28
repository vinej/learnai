"""
05 — Adaptive Conformal Inference (ACI).

Vanilla conformal assumes exchangeability. Markets break that
assumption (regime shifts). ACI (Gibbs & Candès, 2021) updates the
target alpha after each observation:

    alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{y_t in PI_t})

If we missed (y not in PI), increase coverage demand; if we hit, relax.
gamma small (e.g., 0.005) -> slow adaptation; gamma large -> reactive.

Run: python 05_adaptive_conformal.py
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


def aci_loop(model, X, y, alpha_target: float = 0.1, gamma: float = 0.005,
              warmup: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(y)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    alpha_t = alpha_target
    residuals: list[float] = []
    pred = model.predict(X)
    for t in range(n):
        if t < warmup:
            residuals.append(abs(y[t] - pred[t]))
            continue
        # Quantile from the running residual buffer
        q_level = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha_t)) / len(residuals))
        q = float(np.quantile(residuals, q_level))
        lo[t], hi[t] = pred[t] - q, pred[t] + q
        # Online update
        hit = int(lo[t] <= y[t] <= hi[t])
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - (1 - hit)), 1e-3, 0.5))
        residuals.append(abs(y[t] - pred[t]))
    valid = ~np.isnan(lo)
    cov = ((y[valid] >= lo[valid]) & (y[valid] <= hi[valid])).mean()
    return lo, hi, float(cov)


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]

    n = len(df)
    train = df.iloc[:int(n * 0.6)]
    test = df.iloc[int(n * 0.6):]

    booster = lgb.train(
        {"objective": "regression_l2", "learning_rate": 0.03,
         "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
        lgb.Dataset(train[feats], label=train["y"]),
        num_boost_round=300,
    )
    lo, hi, cov = aci_loop(booster, test[feats], test["y"].values,
                            alpha_target=0.1, gamma=0.01, warmup=200)
    width = float(np.nanmean(hi - lo))
    print(f"ACI 90% empirical coverage: {cov:.3f}   width={width:.5f}")
    print("Compare to vanilla split conformal in 03 — ACI keeps coverage")
    print("near 0.9 even when volatility regime shifts.")
