"""
09 — Feature selection

Why bother selecting features?
  • Faster training and inference.
  • Less overfitting (fewer parameters to memorize noise on).
  • Easier to interpret.
  • Cheaper to maintain in production.

Three families:

  FILTER METHODS    — score each feature against the target. Cheap, model-agnostic.
                      Examples: variance, mutual information, correlation, chi².
  WRAPPER METHODS   — repeatedly fit a model with subsets of features.
                      Examples: RFE (recursive feature elimination), forward/backward.
  EMBEDDED METHODS  — selection happens during model fitting itself.
                      Examples: Lasso (zeros out coefficients), tree feature importance.

Run: python 09_feature_selection.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Build a problem where only 5 of 50 features are informative
X, y = make_classification(
    n_samples=2000, n_features=50, n_informative=5, n_redundant=5,
    n_repeated=0, random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)


# ───────────────────────────────────────────────────────────
# 1. FILTER — variance threshold (drop near-constant features)
# ───────────────────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_train_vt = vt.fit_transform(X_train)
print(f"after variance threshold:  {X_train_vt.shape[1]} / {X_train.shape[1]} features")


# ───────────────────────────────────────────────────────────
# 2. FILTER — mutual information (top-K)
# ───────────────────────────────────────────────────────────
# Mutual information measures non-linear dependence between feature and target.
selector = SelectKBest(score_func=mutual_info_classif, k=10).fit(X_train, y_train)
selected_idx = np.where(selector.get_support())[0]
print(f"\ntop-10 by mutual info: features {selected_idx.tolist()}")
print(f"scores (top 5): {sorted(selector.scores_, reverse=True)[:5]}")


# ───────────────────────────────────────────────────────────
# 3. WRAPPER — Recursive Feature Elimination (RFE)
# ───────────────────────────────────────────────────────────
# Train a model, drop the LEAST important feature, repeat. Slow but principled.
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=10).fit(X_train, y_train)
print(f"\nRFE selected: features {np.where(rfe.support_)[0].tolist()}")


# ───────────────────────────────────────────────────────────
# 4. EMBEDDED — Lasso (L1) zeros out features
# ───────────────────────────────────────────────────────────
# Train Lasso, then keep only features with nonzero coefficients.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lasso = Lasso(alpha=0.05).fit(X_train_s, y_train)
nonzero = np.where(np.abs(lasso.coef_) > 1e-3)[0]
print(f"\nLasso selected: {len(nonzero)} features → {nonzero.tolist()}")


# ───────────────────────────────────────────────────────────
# 5. EMBEDDED — tree feature importance
# ───────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_).sort_values(ascending=False)
print(f"\nRandom Forest top 10 by importance: {importances.head(10).index.tolist()}")

# SelectFromModel wraps any estimator with a `feature_importances_` or
# `coef_` and gives you an automatic top-N selector.
sel = SelectFromModel(rf, threshold="median").fit(X_train, y_train)
print(f"SelectFromModel kept {sel.get_support().sum()} features (median importance threshold)")


# ───────────────────────────────────────────────────────────
# Does selection actually help here?
# ───────────────────────────────────────────────────────────
# Compare a model on ALL features vs only the top 10 from mutual info.
all_pipe = LogisticRegression(max_iter=1000).fit(X_train, y_train)
all_acc = accuracy_score(y_test, all_pipe.predict(X_test))

X_train_top = selector.transform(X_train)
X_test_top  = selector.transform(X_test)
sel_pipe = LogisticRegression(max_iter=1000).fit(X_train_top, y_train)
sel_acc = accuracy_score(y_test, sel_pipe.predict(X_test_top))

print(f"\nfull   features ({X_train.shape[1]}): accuracy {all_acc:.3f}")
print(f"top-10 features              : accuracy {sel_acc:.3f}")


# ───────────────────────────────────────────────────────────
# Best practice
# ───────────────────────────────────────────────────────────
# • Do feature selection INSIDE your CV / Pipeline. Otherwise you're using
#   target information from the validation fold to pick features → leakage.
# • Start with simple filters; only escalate to wrappers if you have time.
# • For trees / GBMs, plain feature importance + SelectFromModel often
#   beats elaborate wrappers, at a fraction of the cost.
# • For interpretability, use SHAP — covered in Module 4.6.
