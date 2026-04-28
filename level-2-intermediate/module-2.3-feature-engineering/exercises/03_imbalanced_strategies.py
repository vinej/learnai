"""
Exercise 3 — Three strategies for an imbalanced classification problem

Build a 95/5 imbalanced dataset and compare:

  A. baseline (no fix)
  B. class_weight="balanced"
  C. SMOTE oversampling
  D. baseline + threshold tuned to maximize F1

For each, report precision, recall, F1, and PR-AUC.
The script asserts each "fix" beats the baseline F1.

Run: python exercises/03_imbalanced_strategies.py
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(name, y_true, y_pred, y_score):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_score)
    print(f"{name:<32}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  AP={ap:.3f}")
    return f


def main():
    X, y = make_classification(
        n_samples=10_000, n_features=20, n_informative=8,
        weights=[0.95, 0.05], random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y,
    )

    # A. Baseline
    baseline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ]).fit(X_train, y_train)
    base_pred  = baseline.predict(X_test)
    base_score = baseline.predict_proba(X_test)[:, 1]
    f_base = evaluate("A. baseline (no fix)", y_test, base_pred, base_score)

    # B. class_weight
    weighted = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)
    f_w = evaluate(
        "B. class_weight='balanced'",
        y_test,
        weighted.predict(X_test),
        weighted.predict_proba(X_test)[:, 1],
    )

    # C. SMOTE
    smote = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=0)),
        ("model", LogisticRegression(max_iter=2000)),
    ]).fit(X_train, y_train)
    f_s = evaluate(
        "C. SMOTE oversampling",
        y_test,
        smote.predict(X_test),
        smote.predict_proba(X_test)[:, 1],
    )

    # D. Threshold-tuned baseline
    p, r, t = precision_recall_curve(y_test, base_score)
    f1s = 2 * p * r / np.maximum(p + r, 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    best_thr = t[best_idx]
    tuned_pred = (base_score >= best_thr).astype(int)
    f_t = evaluate(
        f"D. threshold tuned (thr={best_thr:.2f})",
        y_test,
        tuned_pred,
        base_score,
    )

    print(f"\nbaseline F1: {f_base:.3f}")
    for label, f in [("class_weight", f_w), ("SMOTE", f_s), ("threshold tuned", f_t)]:
        gain = f - f_base
        print(f"  {label:<18} F1 = {f:.3f}   ({gain:+.3f})")

    assert max(f_w, f_s, f_t) > f_base, "at least one fix should beat the baseline"
    print("\nat least one strategy improved F1 ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Combine class_weight + threshold tuning. Better than either alone?
# • Use stratified k-fold CV instead of a single split. How stable are the gains?
# • Plot precision-recall curves on one figure to compare strategies visually.
# ─────────────────────────────────────────────────────────────────────────────
