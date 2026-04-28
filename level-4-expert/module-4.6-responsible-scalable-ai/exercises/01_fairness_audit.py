"""
Exercise 1 — Audit a classifier for fairness; mitigate

We train a classifier on a synthetic dataset where a sensitive attribute
correlates with the features. Audit it. Then apply Fairlearn's
ThresholdOptimizer to satisfy equalized odds.

Asserts:
  - Equalized-odds difference DROPS after mitigation.
  - Overall accuracy doesn't collapse (loss < 5%).

Run: python exercises/01_fairness_audit.py
"""

import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a dataset where group correlates with feature 0
# ───────────────────────────────────────────────────────────
n = 6000
X, y = make_classification(n_samples=n, n_features=10,
                            weights=[0.55, 0.45], random_state=0)
group = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)


X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, group, test_size=0.3, random_state=0, stratify=y,
)


# ───────────────────────────────────────────────────────────
# Baseline classifier
# ───────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
pred_baseline = clf.predict(X_test)
acc_baseline = accuracy_score(y_test, pred_baseline)
eod_baseline = equalized_odds_difference(y_test, pred_baseline, sensitive_features=g_test)

frame = MetricFrame(
    metrics={"selection_rate": selection_rate, "TPR": true_positive_rate, "acc": accuracy_score},
    y_true=y_test, y_pred=pred_baseline, sensitive_features=g_test,
)
print("=== baseline ===")
print(frame.by_group.round(3))
print(f"  overall accuracy: {acc_baseline:.3f}")
print(f"  equalized-odds Δ: {eod_baseline:.3f}")


# ───────────────────────────────────────────────────────────
# Mitigation: ThresholdOptimizer (postprocessing)
# ───────────────────────────────────────────────────────────
mitigator = ThresholdOptimizer(
    estimator=clf,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
).fit(X_train, y_train, sensitive_features=g_train)

pred_mit = mitigator.predict(X_test, sensitive_features=g_test)
acc_mit = accuracy_score(y_test, pred_mit)
eod_mit = equalized_odds_difference(y_test, pred_mit, sensitive_features=g_test)


frame_mit = MetricFrame(
    metrics={"selection_rate": selection_rate, "TPR": true_positive_rate, "acc": accuracy_score},
    y_true=y_test, y_pred=pred_mit, sensitive_features=g_test,
)
print("\n=== mitigated ===")
print(frame_mit.by_group.round(3))
print(f"  overall accuracy: {acc_mit:.3f}  (Δ vs baseline {acc_mit - acc_baseline:+.3f})")
print(f"  equalized-odds Δ: {eod_mit:.3f}  (vs baseline {eod_baseline:.3f})")


# ───────────────────────────────────────────────────────────
# Assertions
# ───────────────────────────────────────────────────────────
assert eod_mit < eod_baseline, \
    f"mitigation should REDUCE equalized-odds Δ; got {eod_mit:.3f} vs {eod_baseline:.3f}"
assert acc_mit >= acc_baseline - 0.05, \
    f"accuracy collapsed: {acc_mit:.3f} vs {acc_baseline:.3f}"
print("\nfairness audit + mitigation works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Try `constraints="demographic_parity"` instead. Compare trade-offs.
# • Try ExponentiatedGradient (in-process mitigator). It re-trains the model
#   under a constraint instead of post-hoc threshold tuning.
# • Audit by INTERSECTIONS (e.g., group × age band). Disparities often
#   hide in subgroup intersections that single-axis metrics miss.
# ─────────────────────────────────────────────────────────────────────────────
