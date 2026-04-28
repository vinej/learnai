"""
01 — Window dataset

The fundamental abstraction for sequence forecasting:

    given a sequence x_0, x_1, ..., x_T,
    produce (X_i, y_i) where
      X_i = [x_{i}, x_{i+1}, ..., x_{i+L-1}]      (input window of length L)
      y_i = [x_{i+L}, ..., x_{i+L+H-1}]            (target horizon of H)

We'll also support exogenous features that align with each timestep.

Run: python 01_dataset_window.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from _common import fetch_market


class WindowDataset(Dataset):
    """Sliding window of length L predicting H steps ahead.

    target_col is the column to forecast; feature_cols are exogenous
    inputs included in X but NOT predicted.
    """
    def __init__(self, df: pd.DataFrame, target_col: str,
                  feature_cols: list[str] | None,
                  L: int, H: int):
        feats = [target_col] + (feature_cols or [])
        arr = df[feats].to_numpy(dtype=np.float32)
        targets = df[target_col].to_numpy(dtype=np.float32)
        n = len(arr) - L - H + 1
        if n <= 0:
            raise ValueError(f"series too short for L={L}, H={H}")
        self.X = np.stack([arr[i:i + L] for i in range(n)])
        self.y = np.stack([targets[i + L:i + L + H] for i in range(n)])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = pd.DataFrame({
        "logret": np.log(px / px.shift(1)),
        "abs_ret": (px.pct_change()).abs(),
    }).dropna()

    ds = WindowDataset(df, target_col="logret",
                        feature_cols=["abs_ret"], L=60, H=5)
    print(f"Dataset size: {len(ds)}, X[0]={ds.X[0].shape}, y[0]={ds.y[0].shape}")

    loader = DataLoader(ds, batch_size=32, shuffle=False)
    for xb, yb in loader:
        print("Batch X:", xb.shape, "Y:", yb.shape)
        break

    # CRITICAL: shuffle=False at training time on full series, OR shuffle=True
    # on a chronological train slice (the LATTER is correct — windows within
    # the train slice can be shuffled because each is self-contained).
