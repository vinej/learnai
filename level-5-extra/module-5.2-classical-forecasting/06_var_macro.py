"""
06 — VAR for macro forecasting

Vector AutoRegression generalizes AR(p) to a vector of series:
    y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t

Useful when series are mutually predictive — yields, inflation,
unemployment all influence each other. Granger-causality and
impulse-response functions fall out for free.

Run: python 06_var_macro.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from _common import fetch_fred

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    unrate = fetch_fred("UNRATE").asfreq("MS").ffill()
    dgs10 = fetch_fred("DGS10").resample("MS").mean().ffill()

    df = pd.concat({
        "cpi_yoy":  cpi.pct_change(12) * 100,
        "unrate":   unrate,
        "dgs10":    dgs10,
    }, axis=1).dropna().iloc[-300:]

    train, test = df.iloc[:-12], df.iloc[-12:]

    model = VAR(train)
    # AIC selects p; cap at 12 to keep things sane
    sel = model.select_order(maxlags=12)
    p = int(sel.aic)
    print("Selected lag order p =", p)

    res = model.fit(p)
    print(res.summary().__str__()[:1500], "...")

    # 12-month forecast
    forecast = res.forecast(train.values[-p:], steps=12)
    fc_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)

    print("\n--- Forecast vs actual (last 12 months) ---")
    for col in df.columns:
        mae = (fc_df[col] - test[col]).abs().mean()
        print(f"  {col:<10}  MAE = {mae:.3f}")

    # Granger causality: does CPI YoY help predict the 10Y yield?
    print("\nGranger test: cpi_yoy -> dgs10")
    gc = res.test_causality("dgs10", ["cpi_yoy"], kind="f")
    print(f"  F-stat={gc.test_statistic:.2f}, p={gc.pvalue:.4f}")
