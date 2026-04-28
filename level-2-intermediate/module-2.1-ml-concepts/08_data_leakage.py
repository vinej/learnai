"""
08 — Data leakage

Data leakage is when information from outside the training set sneaks into
the model. Symptoms: training and validation metrics look amazing, but
production performance is much worse.

Two common flavors:
  1. TARGET LEAKAGE
     A feature contains information that wouldn't be available at prediction
     time (e.g., the feature is computed from the target, or from data
     collected after the prediction would be made).
  2. TRAIN-TEST CONTAMINATION
     Information from the test set bleeds into training (e.g., you fit a
     scaler / imputer on the WHOLE dataset before splitting).

This file demonstrates both — and how to fix them with a Pipeline.

Run: python 08_data_leakage.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_breast_cancer(return_X_y=True)


# ═══════════════════════════════════════════════════════════
# 1. Target leakage — adding a feature derived from y
# ═══════════════════════════════════════════════════════════
# We sneak a noisy copy of the LABEL into the features.
rng = np.random.default_rng(0)
leaked_feature = y + rng.normal(0, 0.1, size=y.shape)        # almost-perfect
X_leaky = np.column_stack([X, leaked_feature])

X_train, X_test, y_train, y_test = train_test_split(
    X_leaky, y, random_state=0, stratify=y,
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
print("=== Target leakage demo ===")
print(f"  test accuracy with leaked feature: "
      f"{accuracy_score(y_test, model.predict(X_test)):.3f}")
print("  (looks suspiciously perfect — that's the smell of leakage.)")


# How to spot it:
#   • Feature importances or coefficients: one feature dominates.
#   • Cross-validated metrics are ~100%.
#   • A "too good to be true" gut check.
print(f"  feature 30 coefficient: {model.named_steps['logisticregression'].coef_[0, -1]:.2f}")
print("  one feature with a huge coefficient → check what produced it.\n")


# ═══════════════════════════════════════════════════════════
# 2. Train-test contamination via PRE-SPLIT scaling
# ═══════════════════════════════════════════════════════════
# THE WRONG WAY — fit scaler on the whole dataset before splitting.
# Now train sees information about the test distribution.
scaler = StandardScaler().fit(X)              # 🚩 scaler "saw" all rows
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, random_state=0, stratify=y,
)
bad_model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
print("=== Pre-split scaling demo ===")
print(f"  WRONG (pre-split scaling) accuracy: "
      f"{accuracy_score(y_test, bad_model.predict(X_test)):.3f}")


# THE RIGHT WAY — wrap the scaler in a Pipeline so it's fit ONLY on the
# training fold each time.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y,
)
good_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
good_pipeline.fit(X_train, y_train)
print(f"  RIGHT (Pipeline) accuracy:          "
      f"{accuracy_score(y_test, good_pipeline.predict(X_test)):.3f}")
print("  (the fix often barely changes the number — but in CV it MATTERS.)\n")


# ═══════════════════════════════════════════════════════════
# 3. Leakage in cross-validation
# ═══════════════════════════════════════════════════════════
# Same lesson, different setting. cross_val_score works correctly only when
# the WHOLE preprocessing chain is in the pipeline. The wrong way:
#
#   X_scaled = StandardScaler().fit_transform(X)
#   cross_val_score(LogisticRegression(), X_scaled, y)   # WRONG
#
# The right way:
clean = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = cross_val_score(clean, X, y, cv=5)
print(f"clean CV: mean accuracy {cv.mean():.3f} ± {cv.std():.3f}")


# ═══════════════════════════════════════════════════════════
# Other forms of leakage to watch for
# ═══════════════════════════════════════════════════════════
# • Feature engineered from the FUTURE in time-series (look-ahead).
# • Group leakage: same person/customer in both train and test
#   (use GroupKFold to split by entity).
# • IDs that encode time (created_at) implicitly leaking the target.
# • Imputation values learned from full dataset → use Pipeline.
# • Feature selection done on full dataset → also do it inside the CV loop.
