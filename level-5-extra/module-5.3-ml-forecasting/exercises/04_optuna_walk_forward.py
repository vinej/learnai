"""
Exercise 4 — Optuna study with walk-forward CV and pruning.

Run a 100-trial Optuna study searching over:
  learning_rate, num_leaves, min_data_in_leaf, feature_fraction,
  bagging_fraction, lambda_l2, n_round.

Use TimeSeriesSplit(n_splits=5) inside the objective and report the
best params + a parallel-coordinates plot.

Bonus: enable a pruner (MedianPruner) and report how many trials were
pruned vs completed. Pruning typically saves 30-50% of wall time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: extend 06_optuna_hpo.py with more params + plotting.
