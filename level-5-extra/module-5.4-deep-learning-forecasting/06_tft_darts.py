"""
06 — Temporal Fusion Transformer (TFT) via darts.

TFT (Lim et al., 2021) is purpose-built for multi-horizon forecasting
with mixed exogenous variables:
- STATIC features (e.g., ticker id, sector)
- KNOWN-FUTURE features (e.g., calendar, scheduled events)
- PAST features (e.g., realized vol, macro releases)

It outputs quantile forecasts directly and the variable-selection &
attention weights are interpretable — a real plus.

Run: python 06_tft_darts.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import fetch_fred, fetch_market

warnings.filterwarnings("ignore")

try:
    from darts import TimeSeries
    from darts.models import TFTModel
except ImportError:
    print("Install: pip install darts")
    raise SystemExit(0)


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    dgs10 = fetch_fred("DGS10").asfreq("B").ffill().reindex(spy.index, method="ffill")
    log_spy = np.log(spy)
    df = pd.DataFrame({"y": log_spy, "dgs10": dgs10}).dropna()

    target = TimeSeries.from_series(df["y"])
    past_cov = TimeSeries.from_series(df["dgs10"])

    horizon = 21
    target_train, target_val = target[:-horizon], target[-horizon:]

    model = TFTModel(
        input_chunk_length=126,
        output_chunk_length=horizon,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=64,
        n_epochs=20,
        likelihood=None,            # set to QuantileRegression for intervals
        loss_fn=None,                # default = MSE
        random_state=0,
        add_relative_index=True,
    )
    model.fit(series=target_train, past_covariates=past_cov, verbose=False)
    pred = model.predict(n=horizon, series=target_train, past_covariates=past_cov)

    pred_vals = np.exp(pred.values().flatten())
    actual_vals = np.exp(target_val.values().flatten())
    err = np.abs(pred_vals - actual_vals)
    print(f"TFT 21-day-ahead price MAE: ${err.mean():.2f}")
    print(f"vs SPY level                  ~${actual_vals.mean():.2f}")
