"""
04 — TimeGPT (Nixtla) — paid API.

TimeGPT is a forecasting-as-a-service offering. It is one of very few
foundation models that natively supports exogenous variables and
cross-series finetuning via API.

Set NIXTLA_API_KEY in level-5-extra/.env to run this.

Run: python 04_timegpt_api.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from _common import fetch_fred, fetch_market

load_dotenv()

try:
    from nixtla import NixtlaClient
except ImportError:
    print("Install: pip install nixtla")
    raise SystemExit(0)

API_KEY = os.getenv("NIXTLA_API_KEY")
if not API_KEY:
    print("Set NIXTLA_API_KEY in level-5-extra/.env to run this example.")
    raise SystemExit(0)


if __name__ == "__main__":
    client = NixtlaClient(api_key=API_KEY)

    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    dgs10 = fetch_fred("DGS10").asfreq("B").ffill().reindex(spy.index, method="ffill")

    horizon = 21
    df = pd.DataFrame({
        "ds": spy.index, "y": spy.values, "dgs10": dgs10.values
    }).dropna()
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    # Plain univariate
    fc_uni = client.forecast(
        df=train[["ds", "y"]], h=horizon, freq="B",
        level=[80],
    )
    err_uni = (fc_uni["TimeGPT"].values - test["y"].values)

    # With exogenous
    fc_exo = client.forecast(
        df=train[["ds", "y", "dgs10"]], h=horizon, freq="B",
        X_df=test[["ds", "dgs10"]],         # known-future exogenous
        level=[80],
    )
    err_exo = (fc_exo["TimeGPT"].values - test["y"].values)

    print(f"TimeGPT (univariate)    MAE: ${np.abs(err_uni).mean():.2f}")
    print(f"TimeGPT (with DGS10)    MAE: ${np.abs(err_exo).mean():.2f}")

    cov_uni = ((test["y"].values >= fc_uni["TimeGPT-lo-80"].values) &
                (test["y"].values <= fc_uni["TimeGPT-hi-80"].values)).mean()
    print(f"80% PI coverage (uni):  {cov_uni:.2%}")
