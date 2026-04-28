"""
Exercise 2 — N-HiTS beats ETS on CPI YoY.

For monthly CPI YoY, 12-month horizon, walk-forward across 5 windows:
- Fit ETS(add+add).
- Fit N-HiTS with sensible hyperparams.
- Report mean MAE and per-window MAE.

Goal: show that N-HiTS is at least competitive (often better when the
training history is long enough).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
