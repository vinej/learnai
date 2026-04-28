"""
Exercise 3 — CQR on top of LGB quantile.

Take the LGB quantile model from 5.3 (file 03). Apply CQR (file 04)
with target alpha=0.1 and alpha=0.05.

Check:
- Empirical coverage at the nominal level.
- Width vs raw quantile intervals.
- A scatter of (raw_width, cqr_width) — usually CQR ADDS a small
  uniform pad rather than scaling.

Bonus: implement "locally adaptive" CQR by stratifying the calibration
set by realized vol regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
