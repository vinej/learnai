"""
Exercise 1 — Top features by SHAP.

1. Build a feature matrix with 30+ features for SPY direction.
2. Train an XGB classifier on `y_dir = (y_next > 0)`.
3. Compute SHAP values on the holdout set.
4. Report the 10 features with largest mean(|SHAP|).
5. Plot a SHAP summary chart and a dependence plot for the top feature.
6. Discuss: do these features tell a coherent story (momentum? mean reversion? regime?), or is it noise?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
