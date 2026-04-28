"""
05 — Moirai / Moirai-MoE (Salesforce).

Moirai (2024) is a masked-encoder transformer for any-frequency,
any-distribution time series, trained on the LOTSA corpus (~27B
timesteps). Moirai-MoE adds a mixture-of-experts layer for
context-conditional capacity. Both are open weights.

Run: python 05_moirai.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import fetch_market

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
except ImportError:
    print("Install: pip install uni2ts")
    raise SystemExit(0)


if __name__ == "__main__":
    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()
    horizon = 21
    history = spy.iloc[:-horizon]
    actual = spy.iloc[-horizon:]

    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    model = MoiraiForecast(
        module=module,
        prediction_length=horizon,
        context_length=512,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    target = torch.tensor(history.values, dtype=torch.float32).reshape(1, 1, -1)
    pad = torch.zeros(1, 1, dtype=torch.bool)
    pred = model(
        past_target=target,
        past_observed_target=torch.ones_like(target, dtype=torch.bool),
        past_is_pad=pad.expand(-1, target.shape[-1]),
    )
    samples = pred.squeeze(0).cpu().numpy()    # (n_samples, horizon, 1)
    paths = samples[:, :, 0]

    p10 = np.quantile(paths, 0.1, axis=0)
    p50 = np.quantile(paths, 0.5, axis=0)
    p90 = np.quantile(paths, 0.9, axis=0)

    err = np.abs(p50 - actual.values)
    cov = ((actual.values >= p10) & (actual.values <= p90)).mean()
    print(f"Moirai MAE: ${err.mean():.2f}    80% coverage: {cov:.2%}")
