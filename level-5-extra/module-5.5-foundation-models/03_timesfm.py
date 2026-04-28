"""
03 — TimesFM (Google) zero-shot.

TimesFM is a 200M-parameter decoder-only transformer trained on ~100B
timesteps. It excels on the M-competitions in zero-shot settings.

By 2026 the recommended access path is the HuggingFace `timesfm`
package (Google maintained). Install:
    pip install timesfm

Run: python 03_timesfm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_fred

try:
    import timesfm
except ImportError:
    print("Install: pip install timesfm")
    raise SystemExit(0)


if __name__ == "__main__":
    cpi = fetch_fred("CPIAUCSL").asfreq("MS").ffill()
    yoy = (cpi.pct_change(12).dropna() * 100).iloc[-360:]
    horizon = 12
    history = yoy.iloc[:-horizon]
    actual = yoy.iloc[-horizon:]

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",                   # use "gpu" if available
            per_core_batch_size=32,
            horizon_len=horizon,
            num_layers=50,                   # 200M-param config
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
        ),
    )

    df = pd.DataFrame({
        "unique_id": "cpi_yoy",
        "ds": history.index,
        "y": history.values,
    })
    fc_df = tfm.forecast_on_df(
        inputs=df, freq="MS",
        value_name="y", model_name="TimesFM",
    )
    pred = fc_df.set_index("ds")["TimesFM"].iloc[:horizon]
    pred.index = actual.index

    err = (pred - actual).abs()
    print(f"TimesFM MAE on CPI YoY 12-month forecast: {err.mean():.3f}")
