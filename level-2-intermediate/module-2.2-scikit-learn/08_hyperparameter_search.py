"""
08 — Hyperparameter search: GridSearchCV, RandomizedSearchCV, Optuna

Hyperparameters are the dials you set BEFORE training. Tuning them is
itself a form of training — and a common source of leakage if you peek at
the test set. Always tune with cross-validation on the training data only.

Three approaches, in order of sophistication:

  1. GridSearchCV       — exhaustive search over a discrete grid.
                          Easy, slow, scales poorly with #hyperparams.
  2. RandomizedSearchCV — sample N random combinations from distributions.
                          Same total budget, often finds better configs.
  3. Optuna             — Bayesian / TPE search. State-of-the-art.
                          Adapts: spends more samples in promising regions.

Run: python 08_hyperparameter_search.py
"""

import time

import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from scipy.stats import randint, uniform


# Quiet Optuna's default chatter
optuna.logging.set_verbosity(optuna.logging.WARNING)


X, y = load_breast_cancer(return_X_y=True)


# ───────────────────────────────────────────────────────────
# 1. GridSearchCV
# ───────────────────────────────────────────────────────────
grid = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 5, 10],
    "min_samples_leaf": [1, 5],
}
# Total combos: 2 × 3 × 2 = 12, each with 5 CV folds → 60 fits.
t = time.perf_counter()
gs = GridSearchCV(
    RandomForestClassifier(random_state=0, n_jobs=-1),
    grid, cv=5, scoring="accuracy", n_jobs=-1,
).fit(X, y)
grid_time = time.perf_counter() - t

print(f"GridSearchCV:        best CV acc = {gs.best_score_:.4f}   "
      f"time = {grid_time:.2f}s")
print(f"  best params: {gs.best_params_}")


# ───────────────────────────────────────────────────────────
# 2. RandomizedSearchCV
# ───────────────────────────────────────────────────────────
distributions = {
    "n_estimators":     randint(50, 400),
    "max_depth":        [None, 3, 5, 8, 12, 16],
    "min_samples_leaf": randint(1, 10),
    "max_features":     uniform(0.3, 0.7),    # 0.3 .. 1.0 fraction of features
}
t = time.perf_counter()
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=0, n_jobs=-1),
    distributions, n_iter=20, cv=5, scoring="accuracy",
    n_jobs=-1, random_state=0,
).fit(X, y)
rand_time = time.perf_counter() - t

print(f"RandomizedSearchCV:  best CV acc = {rs.best_score_:.4f}   "
      f"time = {rand_time:.2f}s")
print(f"  best params: {rs.best_params_}")


# ───────────────────────────────────────────────────────────
# 3. Optuna — TPE / Bayesian search
# ───────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
        "max_depth":        trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }
    model = RandomForestClassifier(**params, random_state=0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()


t = time.perf_counter()
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30, show_progress_bar=False)
optuna_time = time.perf_counter() - t

print(f"Optuna (TPE):        best CV acc = {study.best_value:.4f}   "
      f"time = {optuna_time:.2f}s")
print(f"  best params: {study.best_params}")


# ───────────────────────────────────────────────────────────
# When to reach for what
# ───────────────────────────────────────────────────────────
# • 1-2 hyperparameters, small budget        → GridSearchCV.
# • 3-5 hyperparameters, modest budget       → RandomizedSearchCV.
# • Many hyperparameters, big budget,
#   continuous spaces, conditional params    → Optuna.
#
# For very expensive training (deep nets, LLM fine-tuning), Optuna's pruners
# (early-kill bad trials) save a lot of compute. See `optuna.pruners`.


# ───────────────────────────────────────────────────────────
# Don't forget nested CV for honest reporting
# ───────────────────────────────────────────────────────────
# If you're going to publish a number, the "right" thing is NESTED CV:
#   outer CV  → splits data into train+test for honest evaluation
#   inner CV  → searches hyperparams inside each outer training fold
# It's expensive (folds × trials), so most teams report a single best-config
# CV score AND a held-out test number — and live with the slight optimism.
