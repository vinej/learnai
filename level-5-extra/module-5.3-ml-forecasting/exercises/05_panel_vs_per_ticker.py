"""
Exercise 5 — Panel model vs per-ticker models.

Using mlforecast (or by hand):

A) Train ONE LightGBM on a stacked dataset of 5 sector ETFs (XLK, XLF,
   XLE, XLV, XLY). Add a ticker-id feature.
B) Train FIVE LightGBMs, one per ticker.

Walk-forward evaluate both. Where does the panel model help (small
ticker?) and where does the per-ticker beat it (idiosyncratic ETF)?

Hypothesis to test: panel models help most when individual series are
short or noisy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and discuss in 5-10 lines.
