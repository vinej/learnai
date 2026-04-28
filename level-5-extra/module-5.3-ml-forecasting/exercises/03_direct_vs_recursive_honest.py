"""
Exercise 3 — Honest recursive multi-step.

The recursive demo in 04_direct_vs_recursive.py cheated. Build an
honest recursive forecaster:

- Predict r_{t+1} using a 1-step model.
- Append the prediction to the feature window.
- Recompute lag/rolling features INCLUDING the prediction.
- Predict r_{t+2}, etc. through h steps.

Compare to direct at h in {2, 5, 10, 21}. Show the gap grows with h.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement honest recursive loop. mlforecast.predict() does this for free
# if you can't be bothered to roll your own — but try the manual version once.
