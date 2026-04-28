"""
Exercise 4 — End-to-end walk-forward backtest.

Pipeline:
1. SPY daily features from 5.3 file 01.
2. LGB regressor (point) on 5-day cumulative return.
3. CQR-conformal interval at alpha=0.2.
4. Position sizing: kelly-lite using point/sigma_hat from interval width.
5. Daily P&L with 1bp transaction cost.

Walk-forward 5 windows, each 1 year of test, refit per window.

Output:
- Sharpe, Sortino, Max DD, Calmar.
- Equity curve plot.
- Hit rate, expectancy.
- Comparison vs SPY buy-and-hold over the same period.

Honest expectation: with a real signal this barely improves on
buy-and-hold after costs. With a junk signal it's noticeably worse.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
