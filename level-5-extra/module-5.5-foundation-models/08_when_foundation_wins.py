"""
08 — When does a foundation model beat a tuned local model?

A reliable empirical pattern in 2024-2025 (and unlikely to flip by 2026):

    Foundation models WIN when:
      - History is short (< 3 years).
      - The series is one of many similar (panel transfer is happening
        for free — they were trained on similar series).
      - You don't want to do feature engineering.

    Local tuned models WIN when:
      - You have rich exogenous features (macro, news, order flow).
      - You can bake in domain priors (e.g., earnings calendar effects).
      - Compute / latency budget is tight.

This file runs Chronos-Bolt zero-shot vs a tuned LightGBM on three
targets and prints a table.

Run: python 08_when_foundation_wins.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

from _common import fetch_fred, fetch_market

# Re-use the FE module from 5.3
_M53 = pathlib.Path(__file__).resolve().parents[1] / "module-5.3-ml-forecasting"
_spec = importlib.util.spec_from_file_location("fe", _M53 / "01_feature_engineering.py")
fe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fe)


def chronos_forecast(history: pd.Series, h: int) -> pd.Series:
    import torch
    from chronos import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    ctx = torch.tensor(history.values, dtype=torch.float32).unsqueeze(0)
    _, means = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.5])
    return pd.Series(means.cpu().numpy().flatten())


def lgb_forecast(prices: pd.Series, h: int) -> pd.Series:
    import lightgbm as lgb
    df = fe.build_feature_matrix(prices)
    feats = [c for c in df.columns if c not in ("close", "y")]
    train = df.iloc[:-h]
    booster = lgb.train(
        {"objective": "regression_l1", "learning_rate": 0.03,
         "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1},
        lgb.Dataset(train[feats], label=train["y"]),
        num_boost_round=300,
    )
    test = df.iloc[-h:]
    return pd.Series(booster.predict(test[feats]))


if __name__ == "__main__":
    HORIZON = 21
    targets = {
        "SPY level":    fetch_market("SPY")["Close"].asfreq("B").ffill(),
        "BTC level":    fetch_market("BTC-USD")["Close"].asfreq("D").ffill(),
        "CPI YoY":      (fetch_fred("CPIAUCSL").asfreq("MS").ffill().pct_change(12).dropna() * 100),
    }
    rows = []
    for name, s in targets.items():
        h_actual = s.iloc[-HORIZON:]
        # Chronos eats raw level — it's been pretrained on level-shaped data.
        try:
            chronos_pred = chronos_forecast(s.iloc[:-HORIZON], HORIZON)
            chronos_pred.index = h_actual.index
            chronos_mae = float((chronos_pred - h_actual).abs().mean())
        except Exception as e:                # noqa: BLE001
            chronos_mae = float("nan")
            print(f"chronos failed for {name}: {e}")

        # LGB needs price-shaped features; for CPI we adapt by building
        # a lighter feature set. Here we only run for assets:
        if name.endswith("level"):
            try:
                lgb_pred = lgb_forecast(s, HORIZON)
                lgb_pred.index = h_actual.index
                lgb_mae = float((lgb_pred - h_actual).abs().mean())
            except Exception as e:           # noqa: BLE001
                lgb_mae = float("nan")
                print(f"lgb failed for {name}: {e}")
        else:
            lgb_mae = float("nan")
        rows.append({"target": name, "chronos_mae": chronos_mae, "lgb_mae": lgb_mae})

    print("\nMAE on 21-step holdout (lower = better):")
    print(pd.DataFrame(rows).to_string(index=False))
