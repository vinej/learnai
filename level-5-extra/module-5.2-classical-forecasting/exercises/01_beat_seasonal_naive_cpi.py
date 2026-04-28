"""
Exercise 1 — Beat seasonal-naive on monthly CPI YoY.

Build a forecasting workflow that:
- Splits the last 5 years into 12-month walk-forward folds.
- Scores: seasonal-naive(12), ETS(add+add), SARIMA(1,1,1)(1,1,1,12), Prophet.
- Reports mean MAE and rank-by-fold.
- Concludes which is best on this series.

Hint: for low-frequency series, the simpler models often win.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement walk-forward loop with 12-month horizon, 12-month step,
# rolling 10-year training window. Print a final ranking table.
