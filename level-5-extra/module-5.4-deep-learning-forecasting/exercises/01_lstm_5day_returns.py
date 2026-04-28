"""
Exercise 1 — LSTM for 5-day SPY returns.

Build a 2-layer LSTM that takes the last 60 days of (logret, vol21,
distance-to-52w-high) and predicts the cumulative 5-day log return.

Targets:
- Train MAE < 1.5%, val MAE within 10% of zero-prediction baseline.
- Document training curves.
- Discuss whether the model is genuinely learning or just regressing
  to the mean (hint: check predicted distribution vs actual).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement on top of WindowDataset and LSTMForecaster from 02.
