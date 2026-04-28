"""
01 — Forecasting metrics.

  MAE   — robust to outliers, in target units.
  RMSE  — penalizes large errors more.
  MAPE  — percent error; bad when target near zero or negative.
  sMAPE — symmetric MAPE; better behavior near zero, still imperfect.
  MASE  — Mean Absolute Scaled Error; scaled by naive in-sample MAE.
          MASE < 1 means you beat in-sample naive on absolute error.
  RMSSE — Root Mean Squared Scaled Error; squared cousin used in M5.

CRPS and pinball loss live in 02_crps_pinball.py.

Run: python 01_metrics.py
"""
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred): return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))
def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-12)))


def mase(y_true, y_pred, y_train, m: int = 1):
    """Mean Absolute Scaled Error.
    m = seasonal period for the in-sample naive baseline (1 = first-difference).
    """
    y_train = np.asarray(y_train)
    naive_mae = float(np.mean(np.abs(np.diff(y_train, n=m))))
    return mae(y_true, y_pred) / max(naive_mae, 1e-12)


def rmsse(y_true, y_pred, y_train, m: int = 1):
    y_train = np.asarray(y_train)
    naive_mse = float(np.mean(np.diff(y_train, n=m) ** 2))
    return rmse(y_true, y_pred) / max(np.sqrt(naive_mse), 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_train = np.cumsum(rng.normal(0, 1, 200))
    y_test = np.cumsum(rng.normal(0, 1, 50)) + y_train[-1]
    y_pred = y_test + rng.normal(0, 0.5, 50)

    print(f"MAE   = {mae(y_test, y_pred):.4f}")
    print(f"RMSE  = {rmse(y_test, y_pred):.4f}")
    print(f"sMAPE = {smape(y_test, y_pred):.4f}")
    print(f"MASE  = {mase(y_test, y_pred, y_train):.4f}   (<1 beats in-sample naive)")
    print(f"RMSSE = {rmsse(y_test, y_pred, y_train):.4f}")
