"""
Exercise 3 — Reproduce a run from its logged metadata

Step 1: train and log a "reference" run with full metadata (seeds, params).
Step 2: pretend it's a year later. Pull the run's params and seed from MLflow.
Step 3: re-train with ONLY that information and confirm metrics match exactly.

This is the smallest concrete demonstration of why tracking matters.

Run: python exercises/03_reproduce_run.py
"""

import os
import random

import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


EXPERIMENT = "ex3_repro"
mlflow.set_experiment(EXPERIMENT)


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def train(seed: int, n_estimators: int, learning_rate: float, max_depth: int) -> float:
    set_seeds(seed)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
    ).fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# ───────────────────────────────────────────────────────────
# Step 1 — original run, with EVERYTHING logged
# ───────────────────────────────────────────────────────────
original_params = {
    "seed":           7,
    "n_estimators":   150,
    "learning_rate":  0.07,
    "max_depth":      4,
}

with mlflow.start_run(run_name="reference-run") as run:
    mlflow.log_params(original_params)
    score = train(**original_params)
    mlflow.log_metric("test_roc_auc", score)
    reference_run_id = run.info.run_id

print(f"reference run_id = {reference_run_id}")
print(f"original score   = {score:.6f}")


# ───────────────────────────────────────────────────────────
# Step 2 — fetch params back from MLflow
# ───────────────────────────────────────────────────────────
fetched = mlflow.get_run(reference_run_id).data.params
recovered = {
    "seed":          int(fetched["seed"]),
    "n_estimators":  int(fetched["n_estimators"]),
    "learning_rate": float(fetched["learning_rate"]),
    "max_depth":     int(fetched["max_depth"]),
}
print(f"\nrecovered params: {recovered}")


# ───────────────────────────────────────────────────────────
# Step 3 — retrain, confirm exact match
# ───────────────────────────────────────────────────────────
re_score = train(**recovered)
print(f"\nrerun score      = {re_score:.6f}")
print(f"difference       = {abs(score - re_score):.1e}")

assert score == re_score, (
    "scores didn't match — there's an unlogged source of randomness"
)
print("run reproduced exactly ✅")


# ───────────────────────────────────────────────────────────
# What if it didn't match?
# ───────────────────────────────────────────────────────────
# Common culprits:
#   • An n_jobs > 1 estimator with non-deterministic parallelism.
#   • A library version that has changed since the original run.
#   • An unset PYTHONHASHSEED used in dict iteration order.
#   • Non-deterministic GPU operations.
# Identify the culprit by progressively pinning each source of randomness.
