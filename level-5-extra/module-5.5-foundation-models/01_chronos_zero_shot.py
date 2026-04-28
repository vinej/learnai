"""
01 — Zero-shot forecasting with Chronos-Bolt.

Chronos (Amazon, 2024) treats forecasting as a language-modeling
task: it tokenizes a time series into a fixed vocabulary and uses a
T5-style encoder-decoder to "translate" history into future tokens.

Chronos-Bolt (late 2024) is a faster, stronger variant trained on a
larger corpus. By 2026 it's a strong default for zero-shot.

Run: python 01_chronos_zero_shot.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import fetch_market

try:
    from chronos import ChronosBoltPipeline
except ImportError:
    print("Install: pip install chronos-forecasting")
    raise SystemExit(0)


if __name__ == "__main__":
    pipe = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    horizon = 21
    history, actual = spy.iloc[:-horizon], spy.iloc[-horizon:]

    # Chronos-Bolt expects a 1-D tensor of context values per series
    context = torch.tensor(history.values, dtype=torch.float32).unsqueeze(0)
    quantiles, means = pipe.predict_quantiles(
        context=context,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    median = means.cpu().numpy().flatten()
    qs = quantiles.cpu().numpy()[0]   # (H, 3)

    fc = pd.DataFrame({
        "actual": actual.values,
        "p10": qs[:, 0],
        "p50": median,
        "p90": qs[:, 2],
    }, index=actual.index)
    print(fc.round(2))
    err = (fc["actual"] - fc["p50"]).abs()
    print(f"\nChronos-Bolt MAE: ${err.mean():.2f}")
    coverage = ((fc["actual"] >= fc["p10"]) & (fc["actual"] <= fc["p90"])).mean()
    print(f"80% PI coverage:  {coverage:.2%}")
