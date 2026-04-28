"""
04 — N-HiTS for long-horizon forecasting.

N-HiTS (Challu et al., 2023) is N-BEATS with multi-rate downsampling,
giving it explicit multi-scale resolution. It tends to beat N-BEATS at
long horizons (h >= 30) for the same compute.

Run: python 04_nhits_neuralforecast.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_fred

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE
except ImportError:
    print("Install: pip install neuralforecast")
    raise SystemExit(0)


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    yoy = (cpi.pct_change(12).dropna() * 100).iloc[-480:]   # ~40Y

    df = yoy.reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = "cpi_yoy"

    horizon = 24
    train, test = df.iloc[:-horizon], df.iloc[-horizon:]

    nf = NeuralForecast(
        models=[NHITS(
            h=horizon, input_size=72,
            n_blocks=[1, 1, 1],
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[12, 4, 1],
            loss=MAE(),
            max_steps=400,
        )],
        freq="MS",
    )
    nf.fit(train)
    pred = nf.predict().set_index("ds")["NHITS"]
    pred.index = test.set_index("ds").index

    err = (test.set_index("ds")["y"] - pred).abs()
    print(f"N-HiTS MAE on CPI YoY 24-month forecast: {err.mean():.3f}")
