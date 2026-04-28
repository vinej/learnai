"""
07 — Imbalanced classes

When one class is rare (fraud, churn, click-through, disease), the model
naturally learns to predict the majority. Three families of fixes:

  1. CLASS WEIGHTS — tell the model "errors on the rare class hurt more".
                     Free, no extra data manipulation.

  2. RESAMPLING    — change the training distribution.
                     Random over-sampling, SMOTE, ADASYN, random under-sampling.

  3. THRESHOLD TUNING — train normally, then move the decision threshold to
                        trade precision for recall the way YOU want.

Often (1) + (3) is the strongest combination. SMOTE is famous but doesn't
always help — measure on held-out data, not just train metrics.

Run: python 07_imbalanced_classes.py
"""

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ───────────────────────────────────────────────────────────
# 95/5 imbalanced classification
# ───────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=10_000, n_features=20, n_informative=8,
    weights=[0.95, 0.05], random_state=0,
)
print(f"class balance: {np.bincount(y)}    positive rate {y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0,
)


def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    print(f"{name:<26}  prec={precision_score(y_test, y_pred):.3f}  "
          f"rec={recall_score(y_test, y_pred):.3f}  "
          f"F1={f1_score(y_test, y_pred):.3f}  "
          f"AUC={roc_auc_score(y_test, y_score):.3f}  "
          f"AP={average_precision_score(y_test, y_score):.3f}")


# ───────────────────────────────────────────────────────────
# Approach 0 — baseline: do nothing
# ───────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])
evaluate("baseline (no fix)", baseline)


# ───────────────────────────────────────────────────────────
# Approach 1 — class weights
# ───────────────────────────────────────────────────────────
# `class_weight="balanced"` weights inversely proportional to class frequency.
weighted = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
evaluate("class_weight='balanced'", weighted)


# ───────────────────────────────────────────────────────────
# Approach 2 — SMOTE (synthetic over-sampling)
# ───────────────────────────────────────────────────────────
# SMOTE creates synthetic minority samples by interpolating between existing
# minority points. CRUCIAL: only oversample the TRAINING fold, never the test set.
# imbalanced-learn's Pipeline handles this correctly inside CV.
smote = ImbPipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=0)),
    ("model", LogisticRegression(max_iter=2000)),
])
evaluate("SMOTE", smote)


# ───────────────────────────────────────────────────────────
# Approach 3 — ADASYN (focuses on hard minority examples)
# ───────────────────────────────────────────────────────────
adasyn = ImbPipeline([
    ("scale", StandardScaler()),
    ("ada",   ADASYN(random_state=0)),
    ("model", LogisticRegression(max_iter=2000)),
])
evaluate("ADASYN", adasyn)


# ───────────────────────────────────────────────────────────
# Approach 4 — random under-sampling
# ───────────────────────────────────────────────────────────
# Throws away majority samples until classes balance. Cheap; risks losing info.
under = ImbPipeline([
    ("scale", StandardScaler()),
    ("under", RandomUnderSampler(random_state=0)),
    ("model", LogisticRegression(max_iter=2000)),
])
evaluate("random under-sample", under)


# ───────────────────────────────────────────────────────────
# Approach 5 — threshold tuning on top of the baseline
# ───────────────────────────────────────────────────────────
# Train normally, then pick a threshold that meets your business goal.
baseline.fit(X_train, y_train)
y_score = baseline.predict_proba(X_test)[:, 1]

# Find the threshold that gives the best F1
precs, recs, thrs = precision_recall_curve(y_test, y_score)
f1s = 2 * precs * recs / np.maximum(precs + recs, 1e-9)
best_idx = int(np.argmax(f1s[:-1]))
best_thr = thrs[best_idx]
print(f"\nthreshold tuning on baseline:")
print(f"  best F1 threshold = {best_thr:.3f}, F1 = {f1s[best_idx]:.3f}  "
      f"(default 0.5 F1 = {f1_score(y_test, y_score >= 0.5):.3f})")


# ───────────────────────────────────────────────────────────
# Picking what to use
# ───────────────────────────────────────────────────────────
# • Try class_weight FIRST — it's free.
# • Add SMOTE/ADASYN if you have lots of features and the minority class is
#   small but informative.
# • Tune the THRESHOLD to your real business cost (FP vs FN).
# • Always evaluate on the original (imbalanced) test distribution. Never
#   resample the test set.
# • For extreme imbalance (1 in 10,000+), think about anomaly detection
#   methods or specialized losses (focal loss in deep learning).
