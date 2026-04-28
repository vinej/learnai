"""
Exercise 4 — TFT with macro regressors and attention inspection.

Train a TFT on SPY log price, h=21, with:
- past covariates: dgs10, dgs2 (or dff), VIX (yfinance "^VIX")
- known future: dayofweek, month, is_month_end
- (optional) static: sector ETF id if extending to a panel

Then:
- Plot the variable importances over the history.
- Plot a single-window attention pattern across input timesteps.
- Discuss: which variable does the model lean on, and does that change
  in different regimes?

Hint: darts' TFT exposes interpretation via .explain().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and write 5-line conclusion about interpretability.
