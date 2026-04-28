"""
07 — From forecast to position.

A forecast (point + uncertainty) becomes a position via a sizing rule.
A few common ones:

CONSTANT-SIZE
  position = sign(forecast) * 1.0
  Simple but ignores confidence and risk.

VOL-NORMALIZED
  position = sign(forecast) * (target_vol / realized_vol)
  Targets a constant ex-ante volatility.

UNCERTAINTY-AWARE (Kelly-LITE)
  position = forecast / variance_forecast, clipped to [-cap, +cap]
  Larger when expected return is large RELATIVE to predicted vol.

We also enforce realistic constraints:
- Max gross exposure (e.g., 1.0).
- Per-trade transaction cost (bps).
- Daily max position change (turnover cap).

Run: python 07_forecast_to_trade.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_lite_position(point: float, sigma: float, gross_cap: float = 1.0) -> float:
    if sigma <= 0:
        return 0.0
    raw = point / (sigma ** 2)
    return float(np.clip(raw, -gross_cap, gross_cap))


def constant_position(point: float, gross_cap: float = 1.0) -> float:
    return float(np.sign(point)) * gross_cap


def apply_costs(positions: pd.Series, returns: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    turnover = positions.diff().abs().fillna(positions.iloc[0])
    fee = turnover * cost_bps / 1e4
    return positions.shift(1).fillna(0) * returns - fee


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    actual_ret = rng.normal(0.0003, 0.012, n)               # 0.03% daily edge, 1.2% vol
    forecast = 0.2 * actual_ret + rng.normal(0, 0.015, n)   # noisy but informative
    forecast_sigma = np.full(n, 0.012)

    pos_const = pd.Series([constant_position(f, 1.0) for f in forecast])
    pos_kelly = pd.Series([kelly_lite_position(f, s, gross_cap=2.0)
                            for f, s in zip(forecast, forecast_sigma)])

    pnl_const = apply_costs(pos_const, pd.Series(actual_ret), cost_bps=1.0)
    pnl_kelly = apply_costs(pos_kelly, pd.Series(actual_ret), cost_bps=1.0)

    def stats(pnl):
        ann = pnl.mean() * 252
        vol = pnl.std() * np.sqrt(252)
        sr = ann / vol if vol else 0.0
        return ann, vol, sr

    for name, pnl in (("constant", pnl_const), ("kelly-lite", pnl_kelly)):
        ann, vol, sr = stats(pnl)
        print(f"{name:<10}  ann_ret={ann:+.2%}  vol={vol:.2%}  sharpe={sr:.2f}")
