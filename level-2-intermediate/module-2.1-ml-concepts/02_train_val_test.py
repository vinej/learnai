"""
02 — Train / validation / test splits and cross-validation

Why we split:
  TRAIN       — fit the model.
  VALIDATION  — tune hyperparameters / pick the best model.
  TEST        — held out untouched until the very end. The test set tells
                you how the model will perform on NEW data.

THE RULE: never look at the test set during model development. Once you peek
or tune against it, it's no longer an honest estimate.

Run: python 02_train_val_test.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


X, y = load_breast_cancer(return_X_y=True)
print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features, "
      f"class balance: {np.bincount(y)}")


# ───────────────────────────────────────────────────────────
# Simple 80/20 split
# ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0,
)
print(f"\nsimple split: train={len(X_train)}, test={len(X_test)}")


# ───────────────────────────────────────────────────────────
# Stratified split — for imbalanced classes
# ───────────────────────────────────────────────────────────
# `stratify=y` preserves the class proportions in train and test.
# Always do this for classification; otherwise a small rare class can
# end up entirely in one split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)
print(f"\nstratified train balance: {np.bincount(y_train)}")
print(f"stratified test  balance: {np.bincount(y_test)}")


# ───────────────────────────────────────────────────────────
# Three-way split (train / val / test)
# ───────────────────────────────────────────────────────────
# train_test_split twice. First peel off the test, then split the rest.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=0, stratify=y_temp,
)
# Final: 60% train, 20% val, 20% test
print(f"\n3-way split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


# ───────────────────────────────────────────────────────────
# k-fold cross-validation
# ───────────────────────────────────────────────────────────
# Instead of one train/val split, do K of them and average.
# Gives a more reliable estimate, especially on smaller datasets.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n5-fold CV accuracy: mean={cv_scores.mean():.3f}  "
      f"std={cv_scores.std():.3f}  per-fold={cv_scores.round(3).tolist()}")


# Stratified k-fold — same idea but preserves class balance per fold.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print(f"stratified 5-fold accuracy: mean={cv_scores.mean():.3f}  "
      f"std={cv_scores.std():.3f}")


# ───────────────────────────────────────────────────────────
# TimeSeriesSplit — when temporal order MATTERS
# ───────────────────────────────────────────────────────────
# Random shuffling leaks the future into the past. TimeSeriesSplit
# always trains on EARLIER data and tests on LATER data.
ts_splitter = TimeSeriesSplit(n_splits=4)

# Pretend we have a time-ordered dataset
ts_X = np.arange(20).reshape(-1, 1)
print("\nTimeSeriesSplit folds (train indices → test indices):")
for fold, (train_idx, test_idx) in enumerate(ts_splitter.split(ts_X), 1):
    print(f"  fold {fold}: train={train_idx.tolist()}  test={test_idx.tolist()}")


# ───────────────────────────────────────────────────────────
# Common mistakes
# ───────────────────────────────────────────────────────────
# • Re-running train_test_split with no random_state → different splits each
#   time → results aren't reproducible.
# • Calling .fit on the FULL dataset before splitting (e.g. fitting a scaler
#   on X, then splitting). That leaks information about the test set into
#   training. Use a Pipeline so transforms only see training data.
# • Tuning hyperparameters against the test set → "test set rot". Use a
#   validation set or cross-validation; reserve test for the FINAL number.
