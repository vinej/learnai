"""
01 — Why track experiments?

Imagine you've been tuning a model for two weeks. You have a dozen Jupyter
notebooks, scattered print() outputs, and a "best_model.pkl" with no record
of what produced it.

Questions you can't answer:
  • Which hyperparameters gave the best validation score?
  • Which features did that run use?
  • What random seed was set?
  • Which library versions are in the saved model?
  • Does run X really beat run Y, or are they within noise?

THE FIX: log every experiment to a tracking server. Each run captures
parameters, metrics, code version, environment, and any artifacts you want.
You query and compare runs later.

The two heavyweights:
  • MLflow — open-source, local-first, model registry built in.
  • Weights & Biases — managed service, beautiful dashboards, paid for teams.

This module uses MLflow because it runs entirely offline.

Run: python 01_why_track.py
"""

import time

import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ───────────────────────────────────────────────────────────
# What "untracked" looks like
# ───────────────────────────────────────────────────────────
def the_old_way():
    X, y = load_breast_cancer(return_X_y=True)
    Xt, Xv, yt, yv = train_test_split(X, y, random_state=0, stratify=y)

    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=C, max_iter=2000).fit(Xt, yt)
        acc = accuracy_score(yv, model.predict(Xv))
        # We just print to the terminal. Tomorrow we'll forget.
        print(f"C={C}  acc={acc:.3f}")


# ───────────────────────────────────────────────────────────
# The same loop, tracked
# ───────────────────────────────────────────────────────────
def the_new_way():
    mlflow.set_experiment("01_intro")
    X, y = load_breast_cancer(return_X_y=True)
    Xt, Xv, yt, yv = train_test_split(X, y, random_state=0, stratify=y)

    for C in [0.01, 0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"C={C}"):
            mlflow.log_param("C", C)
            mlflow.log_param("solver", "lbfgs")
            mlflow.log_param("max_iter", 2000)

            t0 = time.perf_counter()
            model = LogisticRegression(C=C, max_iter=2000).fit(Xt, yt)
            train_time = time.perf_counter() - t0

            acc = accuracy_score(yv, model.predict(Xv))
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("train_seconds", train_time)


print("=== untracked (chaos) ===")
the_old_way()

print("\n=== tracked with MLflow ===")
the_new_way()

print("\nrun `mlflow ui` in this folder, then open http://127.0.0.1:5000")
print("you'll see one experiment ('01_intro') with 4 runs, each with their")
print("parameters, metrics, and timestamps. Try sorting by val_accuracy.")


# ───────────────────────────────────────────────────────────
# What MLflow records by default
# ───────────────────────────────────────────────────────────
# • params  — anything you pass to log_param()
# • metrics — anything you pass to log_metric() (can be logged at each step)
# • artifacts — files you log (plots, model dumps, configs)
# • tags    — git commit, user, run name, custom labels
# • timestamps — when the run started/ended


# ───────────────────────────────────────────────────────────
# Common patterns we'll cover next
# ───────────────────────────────────────────────────────────
# 02 — manual logging in detail (params/metrics/artifacts/nested runs)
# 03 — autolog: zero-effort tracking for sklearn/xgboost/etc.
# 04 — querying and comparing runs from code
# 05 — the model registry (champion/challenger workflow)
# 06 — reproducibility (seeds, env locks, deterministic flags)
