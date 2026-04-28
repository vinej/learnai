"""
Exercise 5 — Ensemble of foundation models.

Average forecasts from {Chronos-Bolt, TimesFM, Moirai} on a target of
your choice. Compare to:
- Each individual foundation model.
- A 4th-member ensemble that adds a tuned LightGBM.

Three weighting schemes:
- Simple average
- Inverse-MAE weighted (using validation, not test, for honesty)
- Stacking: a Ridge regression that learns to combine forecasts on a
  validation window.

Report MAE on a true holdout. Stacking usually wins on a long-enough
validation window; simple average wins on short windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and write the conclusion.
