"""
02 — Distributional forecasts via path sampling.

Chronos-Bolt outputs quantiles directly, but the original Chronos and
many other foundation models output token distributions. Sampling many
paths and aggregating gives you any quantile (and lets you compute
joint statistics over horizons, like P(SPY drops > 5% in 21 days)).

Run: python 02_chronos_probabilistic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import fetch_market

try:
    from chronos import ChronosPipeline
except ImportError:
    print("Install: pip install chronos-forecasting")
    raise SystemExit(0)


if __name__ == "__main__":
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",      # smaller; CPU-friendly
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    horizon = 21
    history = spy.iloc[:-horizon]
    actual = spy.iloc[-horizon:]

    context = torch.tensor(history.values, dtype=torch.float32).unsqueeze(0)
    n_samples = 200
    samples = pipe.predict(context=context, prediction_length=horizon, num_samples=n_samples)
    paths = samples.cpu().numpy()[0]    # (n_samples, horizon)

    p10 = np.quantile(paths, 0.1, axis=0)
    p50 = np.quantile(paths, 0.5, axis=0)
    p90 = np.quantile(paths, 0.9, axis=0)

    err = np.abs(p50 - actual.values)
    coverage = float(((actual.values >= p10) & (actual.values <= p90)).mean())
    print(f"Chronos (sample-based) MAE: ${err.mean():.2f}")
    print(f"80% PI coverage:            {coverage:.2%}")

    # Joint event probability: P(min over horizon < 0.95 * last)
    last = float(history.iloc[-1])
    cutoff = last * 0.95
    p_drawdown = float((paths.min(axis=1) < cutoff).mean())
    print(f"\nP(any day in next {horizon} hits {cutoff:.0f} = -5%): {p_drawdown:.2%}")
