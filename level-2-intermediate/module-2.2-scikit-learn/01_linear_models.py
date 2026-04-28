"""
01 — Linear models: regression and classification

The simplest, most interpretable, often the strongest baseline.

  LinearRegression                — y = w·x + b. Solves ordinary least squares.
  Ridge / Lasso / ElasticNet      — same, with L2 / L1 / both regularization.
  LogisticRegression              — for classification. Returns probabilities.

The scikit-learn API is consistent across ALL models:
  model.fit(X, y)             — train
  model.predict(X)            — discrete prediction
  model.predict_proba(X)      — class probabilities (classifiers)
  model.score(X, y)           — default metric (R² for reg, accuracy for cls)

Run: python 01_linear_models.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ───────────────────────────────────────────────────────────
# Linear regression family
# ───────────────────────────────────────────────────────────
X, y = fetch_california_housing(return_X_y=True)
feature_names = fetch_california_housing().feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = {
    "LinearRegression":   LinearRegression(),
    "Ridge (α=1.0)":      Ridge(alpha=1.0),
    "Lasso (α=0.1)":      Lasso(alpha=0.1),
    "ElasticNet (α=0.1)": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

print(f"{'model':<22}  {'RMSE':>8}  nonzero coefs")
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m)
    pipe.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe.predict(X_test)) ** 0.5
    coefs = pipe[-1].coef_
    print(f"{name:<22}  {rmse:>8.3f}  {int(np.sum(np.abs(coefs) > 1e-3))}")


# ───────────────────────────────────────────────────────────
# Inspect coefficients (interpretability)
# ───────────────────────────────────────────────────────────
# Coefficients show which features push the prediction up or down.
# After StandardScaler, magnitudes are comparable across features.
linreg = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
coef_table = pd.Series(linreg[-1].coef_, index=feature_names).sort_values()
print("\nlinear coefficients (after standardization):")
print(coef_table.round(3))


# ───────────────────────────────────────────────────────────
# Logistic regression for classification
# ───────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0,
)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)
print(f"\nIris accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")

# Probabilities for the first 3 test samples
proba = clf.predict_proba(X_test[:3])
print("\nprobabilities for first 3 samples:")
print(pd.DataFrame(proba, columns=iris.target_names).round(3))


# ───────────────────────────────────────────────────────────
# Multi-class strategies
# ───────────────────────────────────────────────────────────
# • multinomial (default in modern sklearn): one model, softmax output
# • one-vs-rest: K binary classifiers, one per class
# Some classifiers (e.g. SVM) don't natively support multi-class — sklearn
# wraps them with one-vs-rest automatically.


# ───────────────────────────────────────────────────────────
# When to use linear models
# ───────────────────────────────────────────────────────────
# • You need INTERPRETABILITY — coefficients are easy to explain.
# • The problem is roughly linear in the features (or you can engineer it so).
# • You have lots of features and not many rows (regularization shines).
# • You want a fast, reliable baseline before complex models.
