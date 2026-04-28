"""
03 — ARIMA / SARIMA

ARIMA(p, d, q):
  d = order of differencing to achieve stationarity
  p = AR order (use PACF cutoff to pick)
  q = MA order (use ACF cutoff to pick)

SARIMA(p,d,q)(P,D,Q,m):
  uppercase = seasonal counterparts; m = season length

In practice, AutoARIMA from statsforecast is faster and saner than
hand-tuning. We'll show both — manual identification first to build
intuition, then auto.

Run: python 03_arima_sarima.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from _common import fetch_fred

warnings.filterwarnings("ignore")


def manual_sarima(train: pd.Series, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12)) -> ARIMA:
    return ARIMA(train, order=order, seasonal_order=seasonal_order).fit()


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill().iloc[-360:]   # 30Y
    log_cpi = np.log(cpi)

    train, test = log_cpi.iloc[:-24], log_cpi.iloc[-24:]

    # Manual SARIMA(1,1,1)(1,1,1,12) on log CPI
    fit = manual_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    print(fit.summary().tables[0])
    fc = fit.forecast(24)
    fc.index = test.index
    err = (np.exp(test) - np.exp(fc)).abs()
    print(f"\nManual SARIMA MAE on CPI level: {err.mean():.3f}")

    # Try AutoARIMA via statsforecast (uses Hyndman-Khandakar algorithm)
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        df = train.reset_index().rename(columns={train.index.name or "index": "ds"})
        df.columns = ["ds", "y"]
        df["unique_id"] = "cpi"
        sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
        sf.fit(df)
        h = 24
        pred = sf.predict(h=h)
        pred.index = test.index
        auto_fc = pred["AutoARIMA"]
        err2 = (np.exp(test) - np.exp(auto_fc)).abs()
        print(f"AutoARIMA   MAE on CPI level: {err2.mean():.3f}")
    except ImportError:
        print("(statsforecast not installed — skipping AutoARIMA)")
