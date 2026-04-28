"""
05 — Classification metrics

Accuracy is the metric beginners reach for first. It's also the one that lies
to you most often, especially with imbalanced classes.

This file walks through:
  • Accuracy (and why it can be misleading)
  • Precision, Recall, F1
  • Confusion matrix
  • ROC curve / ROC-AUC
  • Precision-recall curve / PR-AUC
  • log-loss

Run: python 05_classification_metrics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ───────────────────────────────────────────────────────────
# Build an imbalanced classification problem
# ───────────────────────────────────────────────────────────
# 95% negatives, 5% positives — like fraud detection.
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=8,
    weights=[0.95, 0.05], random_state=0,
)
print(f"class balance: {np.bincount(y)}    (positive rate {y.mean():.1%})")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y,
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)                    # hard labels (0 or 1)
y_score = model.predict_proba(X_test)[:, 1]        # P(class = 1)


# ───────────────────────────────────────────────────────────
# Why accuracy lies on imbalanced data
# ───────────────────────────────────────────────────────────
trivial_pred = np.zeros_like(y_test)               # always predict "negative"
print(f"\ntrivial 'always negative' accuracy: {accuracy_score(y_test, trivial_pred):.3f}")
print(f"our model's accuracy:               {accuracy_score(y_test, y_pred):.3f}")
print("→ accuracy alone says they're similar. They are NOT.")


# ───────────────────────────────────────────────────────────
# Confusion matrix
# ───────────────────────────────────────────────────────────
#                    PREDICTED
#                  Neg     Pos
# ACTUAL  Neg  | TN   |  FP    (FP = false alarm)
#         Pos  | FN   |  TP    (FN = missed detection)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nconfusion matrix:\n{cm}")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")


# ───────────────────────────────────────────────────────────
# Precision, recall, F1
# ───────────────────────────────────────────────────────────
# Precision = TP / (TP + FP)   "of those I predicted positive, how many were?"
# Recall    = TP / (TP + FN)   "of all the actual positives, how many did I catch?"
# F1        = harmonic mean of the two.
print(f"\nprecision: {precision_score(y_test, y_pred):.3f}")
print(f"recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")

# Precision vs recall is a TRADEOFF.
#   • Spam filter? You want high PRECISION (few false positives).
#   • Cancer screening? You want high RECALL (don't miss any).


# ───────────────────────────────────────────────────────────
# ROC-AUC and PR-AUC — threshold-free metrics
# ───────────────────────────────────────────────────────────
# Both consider ALL possible decision thresholds.
#   ROC-AUC: how well the model RANKS positives above negatives.
#            0.5 = random, 1.0 = perfect.
#            Can be MISLEADING on heavily imbalanced data — PR-AUC is better there.
#   PR-AUC (average_precision_score): area under the precision-recall curve.
#            Sensitive to the positive class size.
print(f"\nROC-AUC: {roc_auc_score(y_test, y_score):.3f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_score):.3f}")
print(f"log-loss: {log_loss(y_test, y_score):.3f}")


# ───────────────────────────────────────────────────────────
# Plot it all
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=axes[0])
axes[0].set_title("Confusion matrix")
axes[0].set_xlabel("predicted")
axes[0].set_ylabel("actual")

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_title("ROC curve")

# PR curve
PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=axes[2])
axes[2].set_title("Precision-recall curve")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "05_classification_metrics.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '05_classification_metrics.png'}")


# ───────────────────────────────────────────────────────────
# Adjusting the decision threshold
# ───────────────────────────────────────────────────────────
# By default, predict() uses 0.5. You can pick a different threshold to
# trade precision for recall (or vice versa).
precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
# Find the threshold that gives at least 0.85 precision
mask = precisions[:-1] >= 0.85
if mask.any():
    idx = np.argmax(recalls[:-1][mask])
    best_threshold = thresholds[mask][idx]
    print(f"\nfor ≥85% precision, use threshold {best_threshold:.2f} "
          f"(recall = {recalls[:-1][mask][idx]:.2f})")
