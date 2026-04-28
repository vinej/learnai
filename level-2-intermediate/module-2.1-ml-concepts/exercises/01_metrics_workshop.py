"""
Exercise 1 — Metrics workshop

Three classifiers were trained on the SAME imbalanced dataset (5% positive
rate). Their predictions on the test set are below.

For each, compute:
  • accuracy
  • precision, recall, F1
  • ROC-AUC (using the probability scores, not the hard labels)
  • PR-AUC (a.k.a. average_precision_score)

Then ANSWER (in comments at the end) which model you would pick for:
  (a) a fraud-detection system that triggers a manual review on positives
  (b) a free trial → paid conversion predictor (every false positive wastes
      a marketing email; every false negative is a missed sale)

Run: python exercises/01_metrics_workshop.py
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


rng = np.random.default_rng(42)
N = 1000
positive_rate = 0.05
y_true = (rng.random(N) < positive_rate).astype(int)
n_pos = int(y_true.sum())
print(f"true positives in test set: {n_pos} / {N}\n")


# ───────────────────────────────────────────────────────────
# Three pretend models with different behavior
# ───────────────────────────────────────────────────────────
# Model A — the "always negative" baseline (high accuracy but useless)
y_score_A = np.full(N, 0.05)
y_pred_A  = (y_score_A > 0.5).astype(int)

# Model B — high recall, low precision (sensitive but trigger-happy)
y_score_B = np.where(y_true == 1, rng.uniform(0.4, 0.95, N), rng.uniform(0.0, 0.7, N))
y_pred_B  = (y_score_B > 0.4).astype(int)

# Model C — high precision, lower recall (cautious but accurate when it fires)
y_score_C = np.where(y_true == 1, rng.uniform(0.3, 0.95, N), rng.uniform(0.0, 0.4, N))
y_pred_C  = (y_score_C > 0.6).astype(int)


def evaluate(name: str, y_pred: np.ndarray, y_score: np.ndarray):
    print(f"{name}")
    print(f"  accuracy  = {accuracy_score(y_true, y_pred):.3f}")
    # zero_division=0 keeps it quiet when the model never predicts positive
    print(f"  precision = {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  recall    = {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1        = {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  ROC-AUC   = {roc_auc_score(y_true, y_score):.3f}")
    print(f"  PR-AUC    = {average_precision_score(y_true, y_score):.3f}\n")


evaluate("MODEL A (always negative)", y_pred_A, y_score_A)
evaluate("MODEL B (high recall)",     y_pred_B, y_score_B)
evaluate("MODEL C (high precision)",  y_pred_C, y_score_C)


# ───────────────────────────────────────────────────────────
# Your answers
# ───────────────────────────────────────────────────────────
# Fill these in based on the metrics above.
#
# (a) Fraud detection (manual review on positives) → MODEL ___
#     because: ____________________________________
#
# (b) Free-trial conversion predictor → MODEL ___
#     because: ____________________________________
#
# Hint: in (a) you can afford to flag a few extra cases — humans review them.
# In (b) every false positive wastes outreach effort.
