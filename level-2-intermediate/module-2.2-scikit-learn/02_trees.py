"""
02 — Decision trees, random forests, gradient boosting (sklearn-native)

DECISION TREE
  Splits the feature space recursively to minimize impurity.
  Pros: interpretable, handles non-linear, no scaling needed.
  Cons: high variance — a small data change → a very different tree.

RANDOM FOREST
  Many trees, each on a bootstrap sample with a random feature subset.
  Averaging tames the variance. Strong default for tabular data.

GRADIENT BOOSTING (sklearn's HistGradientBoostingRegressor/Classifier)
  Trees built sequentially, each correcting the previous one's residuals.
  Often THE strongest tabular model. XGBoost / LightGBM are faster
  alternatives — covered in the next file.

Run: python 02_trees.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0,
)


# ───────────────────────────────────────────────────────────
# 1. Decision tree — single tree, easy to read
# ───────────────────────────────────────────────────────────
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
print(f"single tree (depth=3) accuracy: {accuracy_score(y_test, tree.predict(X_test)):.3f}")

print("\ntree structure (first few levels):")
print(export_text(tree, feature_names=list(feature_names), max_depth=3))


# ───────────────────────────────────────────────────────────
# 2. Random forest
# ───────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,         # let trees grow fully (forest averages anyway)
    n_jobs=-1,
    random_state=0,
).fit(X_train, y_train)
print(f"\nrandom forest accuracy:        {accuracy_score(y_test, rf.predict(X_test)):.3f}")


# Feature importances (mean decrease in impurity)
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\ntop 10 features (RF importance):")
print(importances.head(10).round(3))


# ───────────────────────────────────────────────────────────
# 3. Gradient boosting (classic & histogram-based)
# ───────────────────────────────────────────────────────────
gbm = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0,
).fit(X_train, y_train)
print(f"\ngradient boosting accuracy:    {accuracy_score(y_test, gbm.predict(X_test)):.3f}")


# HistGradientBoosting — sklearn's fast histogram-based GBM. Handles NaN
# natively and scales much better. Usually first choice in plain sklearn.
hgbm = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.05, random_state=0,
).fit(X_train, y_train)
print(f"hist gradient boosting:        {accuracy_score(y_test, hgbm.predict(X_test)):.3f}")


# ───────────────────────────────────────────────────────────
# Tuning knobs to know
# ───────────────────────────────────────────────────────────
# Random Forest:
#   n_estimators        — more trees → diminishing returns; 100-500 typical
#   max_depth           — None = grow fully; limit to fight overfit if needed
#   min_samples_leaf    — increase to smooth predictions
#   max_features        — random subset per split; 'sqrt' is the classic default
#
# Gradient Boosting:
#   n_estimators        — more rounds = more capacity; pair with early stopping
#   learning_rate       — smaller + more rounds usually wins
#   max_depth           — 3-8 typical; deeper trees = more interactions
#   subsample           — < 1.0 enables stochastic GBM (regularization)


# ───────────────────────────────────────────────────────────
# Pros and cons of trees vs linear
# ───────────────────────────────────────────────────────────
# • Trees handle non-linearity & feature interactions natively.
# • Trees don't need scaling; mostly indifferent to feature distribution.
# • Linear models extrapolate; tree predictions are bounded by training data.
# • Single trees are interpretable; ensembles need SHAP/permutation importance.

print("\n" + classification_report(y_test, hgbm.predict(X_test),
                                   target_names=["malignant", "benign"]))
