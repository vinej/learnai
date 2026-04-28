"""
07 — Pinball (quantile) loss for direct interval forecasts.

Pinball loss at level q:
    L_q(y, y_hat) = max(q*(y - y_hat), (q-1)*(y - y_hat))

Train one head per quantile (or one model with multiple outputs and a
sum of pinball losses). Use neuralforecast's built-in quantile loss
or roll your own:

Run: python 07_quantile_loss.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from _common import fetch_market

_spec = importlib.util.spec_from_file_location(
    "ds", pathlib.Path(__file__).parent / "01_dataset_window.py"
)
ds_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds_mod)
WindowDataset = ds_mod.WindowDataset


QUANTILES = (0.1, 0.5, 0.9)


class QuantileLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, horizon: int, n_q: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.heads = nn.Linear(hidden, horizon * n_q)
        self.horizon = horizon
        self.n_q = n_q

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        # (B, horizon, n_q)
        return self.heads(last).view(-1, self.horizon, self.n_q)


def pinball_loss(pred, target, quantiles=QUANTILES):
    # pred: (B, H, Q), target: (B, H)
    target = target.unsqueeze(-1)              # (B, H, 1)
    err = target - pred                         # (B, H, Q)
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    return torch.maximum(q * err, (q - 1) * err).mean()


if __name__ == "__main__":
    L, H = 60, 5
    px = fetch_market("SPY")["Close"]
    df = pd.DataFrame({
        "logret": np.log(px / px.shift(1)),
        "vol21": np.log(px / px.shift(1)).rolling(21).std().shift(1),
    }).dropna()
    full = WindowDataset(df, target_col="logret", feature_cols=["vol21"], L=L, H=H)
    n = len(full)
    train_ds = Subset(full, list(range(int(n * 0.7))))
    test_ds = Subset(full, list(range(int(n * 0.85), n)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QuantileLSTM(2, 64, H, len(QUANTILES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    for epoch in range(15):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = pinball_loss(model(x), y)
            loss.backward(); opt.step()
            total += float(loss.item()) * len(x)
        print(f"epoch {epoch:>2}  pinball={total/len(train_ds):.5f}")

    # Coverage check on test
    model.eval()
    inside = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)             # (B, H, Q)
            lo, hi = p[..., 0], p[..., -1]
            inside += int(((y >= lo) & (y <= hi)).sum().item())
            total += y.numel()
    print(f"\n80% PI empirical coverage: {inside / total:.3f}    (target = 0.80)")
