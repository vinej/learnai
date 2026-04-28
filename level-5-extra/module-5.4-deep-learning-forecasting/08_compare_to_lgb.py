"""
08 — Head-to-head: deep learning vs LightGBM.

Same target (1-day SPY log return), same feature pool, same splits.
Train:
- LightGBM (from 5.3)
- LSTM (from 02)
- N-BEATS (neuralforecast)
- PatchTST (neuralforecast)

Report MAE, RMSE, and direction accuracy. The expected outcome on
single-asset daily returns: all four are within noise of each other,
LGB is fastest. The DL models earn their keep on:
- Panel data (many similar series)
- Long horizons
- When you need built-in attention/regressor handling

Run: python 08_compare_to_lgb.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import fetch_market


def evaluate(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "model": name,
        "mae": float(np.abs(pred - actual).mean()),
        "rmse": float(np.sqrt(((pred - actual) ** 2).mean())),
        "dir_acc": float((np.sign(pred) == np.sign(actual)).mean()),
    }


if __name__ == "__main__":
    # NOTE: This file is a SCAFFOLD. Each model lives in its own file
    # because their training loops are very different. Implement here:
    #
    # 1. Build long-format dataframe of SPY daily log returns.
    # 2. Run each of (LGB, LSTM, N-BEATS, PatchTST) using the same
    #    train/test split.
    # 3. Collect per-model dicts via evaluate().
    # 4. Print a markdown-style results table.
    #
    # See exercises/05_repro_compare.py for a full working example.
    print("This file is a scaffold; see the exercise version.")
