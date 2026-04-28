"""
Exercise 5 — Walk-forward baseline.

Implement a "naive" forecaster that predicts r_{t+1} = r_t (random walk
on returns; zero is also a valid baseline since E[r] is tiny).
Score it via walk-forward CV with:
- expanding training window starting at 5 years
- 1-quarter test horizon
- 1-quarter step

Report MAE, RMSE, and direction-accuracy averaged across folds.

This baseline is what every fancier model in 5.2-5.5 must beat.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: Reuse walk_forward_splits() from 06_splits_and_leakage.py.
# TODO: predictions = previous return (or zero).
# TODO: print mean(MAE), mean(RMSE), mean(direction_accuracy) +- std.
