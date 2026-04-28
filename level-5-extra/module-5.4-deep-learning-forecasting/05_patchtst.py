"""
05 — PatchTST: a strong transformer baseline for time series.

PatchTST (Nie et al., 2023, "A Time Series is Worth 64 Words")
broke the conventional wisdom that transformers underperform on TS.
Two ideas:
- PATCHING: split the input into non-overlapping patches and treat
  each patch as a token. Cuts attention cost; each token carries more
  semantic content than a single timestep.
- CHANNEL INDEPENDENCE: process each time series independently with
  shared weights, even in multivariate settings. Surprisingly effective.

Train it on a panel of ETFs and see how it stacks up.

Run: python 05_patchtst.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_market

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import MAE
except ImportError:
    print("Install: pip install neuralforecast")
    raise SystemExit(0)


def to_long(prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for t, s in prices.items():
        r = np.log(s / s.shift(1)).dropna()
        rows.append(pd.DataFrame({"unique_id": t, "ds": r.index, "y": r.values}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    prices = {t: fetch_market(t)["Close"] for t in ("SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP")}
    df = to_long(prices)

    horizon = 5
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()

    nf = NeuralForecast(
        models=[PatchTST(
            h=horizon, input_size=128,
            patch_len=16, stride=8,
            n_heads=4, encoder_layers=3,
            hidden_size=128, linear_hidden_size=256,
            loss=MAE(),
            max_steps=300,
        )],
        freq="B",
    )
    nf.fit(train)
    pred = nf.predict()
    print("PatchTST 5-day-ahead return forecasts (most recent):")
    print(pred.pivot(index="ds", columns="unique_id", values="PatchTST").round(5))

    # Cross-validation across the panel
    cv = nf.cross_validation(df=df, n_windows=3, step_size=63)
    cv["abs_err"] = (cv["y"] - cv["PatchTST"]).abs()
    print("\nMean MAE per ticker:")
    print(cv.groupby("unique_id")["abs_err"].mean().round(5))
