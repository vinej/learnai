"""
06 — Lag-Llama (ServiceNow + Mila).

Lag-Llama (2024) is a decoder-only transformer that conditions on
explicit lag features rather than raw history — making it surprisingly
strong on long-context probabilistic forecasting.

This is heavier than the others: needs ~2 GB VRAM and ~30s for a
500-step forecast on CPU. Reduce `context_length` to test quickly.

Run: python 06_lag_llama.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import fetch_market

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
except ImportError:
    print("Install: pip install git+https://github.com/time-series-foundation-models/lag-llama.git")
    raise SystemExit(0)


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    horizon = 21
    history = spy.iloc[:-horizon]

    # See the official notebook — Lag-Llama uses GluonTS-style datasets.
    # For brevity here we sketch the call.
    print("This file demonstrates the Lag-Llama API.")
    print("Full reproducible example: https://github.com/time-series-foundation-models/lag-llama/blob/main/lagllama_zero_shot.ipynb")
    print("Approximate workflow:")
    print("  1. Wrap history as a GluonTS PandasDataset.")
    print("  2. estimator = LagLlamaEstimator(ckpt_path=..., prediction_length=21, context_length=256, ...)")
    print("  3. predictor = estimator.create_predictor(...)")
    print("  4. forecast_it = predictor.predict(test_dataset)")
    print("  5. samples = next(forecast_it).samples   # (n_samples, horizon)")
