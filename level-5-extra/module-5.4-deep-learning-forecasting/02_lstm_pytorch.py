"""
02 — LSTM baseline (PyTorch)

A single-layer LSTM with a linear head. Predict the next 5 days of
log returns from the previous 60 days. Goal: demonstrate the loop and
get a working baseline. Don't expect miracles — daily equity returns
are mostly noise, and an LSTM on raw returns will rarely beat naive.

Run: python 02_lstm_pytorch.py
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

# Load WindowDataset from 01 (digit-prefix module trick)
_spec = importlib.util.spec_from_file_location(
    "ds", pathlib.Path(__file__).parent / "01_dataset_window.py"
)
ds_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds_mod)
WindowDataset = ds_mod.WindowDataset


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int, horizon: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                    # x: (B, L, F)
        out, _ = self.lstm(x)                # (B, L, H)
        last = out[:, -1, :]                 # last hidden state
        return self.head(last)               # (B, horizon)


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        total += float(loss.item()) * len(x); n += len(x)
    return total / n


if __name__ == "__main__":
    L, H = 60, 5
    px = fetch_market("SPY")["Close"]
    df = pd.DataFrame({
        "logret": np.log(px / px.shift(1)),
        "vol21": np.log(px / px.shift(1)).rolling(21).std().shift(1),
    }).dropna()

    full = WindowDataset(df, target_col="logret", feature_cols=["vol21"], L=L, H=H)
    n = len(full)
    train_idx = list(range(int(n * 0.7)))
    val_idx = list(range(int(n * 0.7), int(n * 0.85)))
    test_idx = list(range(int(n * 0.85), n))
    train_ds, val_ds, test_ds = Subset(full, train_idx), Subset(full, val_idx), Subset(full, test_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMForecaster(n_features=2, hidden=64, horizon=H, num_layers=2, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    best_val = float("inf"); patience = 5; bad = 0
    for epoch in range(30):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:>2}  train={tr:.5f}  val={va:.5f}")
        if va < best_val:
            best_val, bad = va, 0
        else:
            bad += 1
            if bad >= patience:
                print("early stop")
                break

    te = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest SmoothL1: {te:.5f}")
