"""
07 — Always start with a baseline

Before any fancy model, build a STUPID one. Why?

  1. It validates your evaluation pipeline (a buggy pipeline can make any
     model look great).
  2. It tells you what "good" looks like. If your sophisticated model only
     beats the baseline by 0.5%, you have a problem.
  3. It's free insurance against complexity.

scikit-learn ships `DummyClassifier` and `DummyRegressor` for exactly this.

Run: python 07_baselines.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ───────────────────────────────────────────────────────────
# Classification baselines
# ───────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)

print("=== Classification baselines (breast cancer) ===\n")

baselines = [
    DummyClassifier(strategy="most_frequent"),     # always predict the majority
    DummyClassifier(strategy="stratified", random_state=0),  # match class proportions
    DummyClassifier(strategy="prior"),             # like most_frequent but supports proba
    DummyClassifier(strategy="uniform", random_state=0),     # 50/50
]

for b in baselines:
    b.fit(X_train, y_train)
    y_pred = b.predict(X_test)
    y_proba = b.predict_proba(X_test)[:, 1]
    print(f"  {b.strategy:<14}  accuracy={accuracy_score(y_test, y_pred):.3f}  "
          f"F1={f1_score(y_test, y_pred):.3f}  "
          f"ROC-AUC={roc_auc_score(y_test, y_proba):.3f}")

# A real model
real = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
y_pred = real.predict(X_test)
y_proba = real.predict_proba(X_test)[:, 1]
print(f"  {'real model':<14}  accuracy={accuracy_score(y_test, y_pred):.3f}  "
      f"F1={f1_score(y_test, y_pred):.3f}  "
      f"ROC-AUC={roc_auc_score(y_test, y_proba):.3f}")


# ───────────────────────────────────────────────────────────
# Regression baselines
# ───────────────────────────────────────────────────────────
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("\n=== Regression baselines (California housing) ===\n")

baselines = [
    DummyRegressor(strategy="mean"),
    DummyRegressor(strategy="median"),
    DummyRegressor(strategy="quantile", quantile=0.9),
]
for b in baselines:
    b.fit(X_train, y_train)
    pred = b.predict(X_test)
    print(f"  {b.strategy:<10}  RMSE={mean_squared_error(y_test, pred) ** 0.5:.3f}  "
          f"MAE={mean_absolute_error(y_test, pred):.3f}")

real = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
pred = real.predict(X_test)
print(f"  {'real model':<10}  RMSE={mean_squared_error(y_test, pred) ** 0.5:.3f}  "
      f"MAE={mean_absolute_error(y_test, pred):.3f}")


# ───────────────────────────────────────────────────────────
# Domain-specific baselines (often more useful)
# ───────────────────────────────────────────────────────────
# Built-in dummies are CLASS-PRIOR baselines. Often a more meaningful
# baseline is domain-aware:
#   • Time series: predict yesterday's value, or the seasonal naive.
#   • Recommender: most-popular items.
#   • Text classifier: regex / keyword rules.
#   • Anomaly detection: sample mean ± 3σ.
# Beat THAT and you've built something useful.
