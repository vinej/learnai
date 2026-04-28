"""
03 — N-BEATS via neuralforecast.

N-BEATS (Oreshkin et al., 2020) is a stack of fully-connected residual
blocks specialized into trend and seasonality basis functions. It's
simple, fast, and surprisingly competitive — it won M4 in 2020 and
remains a strong baseline.

We use Nixtla's `neuralforecast` which implements N-BEATS, N-HiTS,
PatchTST, TFT, and others under one API.

Run: python 03_nbeats_neuralforecast.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_fred

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.losses.pytorch import MAE
except ImportError:
    print("Install: pip install neuralforecast")
    raise SystemExit(0)


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    yoy = (cpi.pct_change(12).dropna() * 100).iloc[-360:]   # 30Y monthly

    df = yoy.reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = "cpi_yoy"

    train = df.iloc[:-12]
    test = df.iloc[-12:]

    nf = NeuralForecast(
        models=[NBEATS(
            h=12, input_size=48,
            stack_types=["identity", "identity"],
            n_blocks=[3, 3],
            mlp_units=[[256, 256], [256, 256]],
            loss=MAE(),
            max_steps=300,
        )],
        freq="MS",
    )
    nf.fit(train)
    pred = nf.predict()
    pred = pred.set_index("ds")["NBEATS"]
    pred.index = test.set_index("ds").index

    err = (test.set_index("ds")["y"] - pred).abs()
    print(f"N-BEATS MAE on CPI YoY 12-month forecast: {err.mean():.3f}")
    out = pd.DataFrame({"actual": test.set_index("ds")["y"], "nbeats": pred}).round(3)
    print(out)
