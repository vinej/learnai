"""
Exercise 1 — Build a clean training dataset.

Goal: produce a single Parquet file at `data_cache/dataset.parquet`
with columns:
    spy, qqq, btc, eurusd, dgs10, cpi_yoy

Daily frequency, business-day index, no NaNs after the warmup period.
CPI must be lagged by 30 days for release timing.

Bonus: extend to also include XLK (tech) and XLE (energy).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: Use _common.fetch_market and _common.fetch_fred to assemble the frame.
# TODO: Resample crypto to business days to align with stocks (or keep all
#       calendar days and choose a strategy — document your choice).
# TODO: Use forward-fill with care; never bfill.
# TODO: Save to data_cache/dataset.parquet and print the head + shape.
