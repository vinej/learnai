"""
01 — Bias and fairness

A model can be ACCURATE OVERALL but unfair across subgroups. Fairness
analysis asks: does the model behave equitably across groups defined
by a SENSITIVE ATTRIBUTE (gender, race, age band, geography)?

Three common metrics (binary classifier with predictions Ŷ ∈ {0, 1}):

  DEMOGRAPHIC PARITY
    P(Ŷ=1 | group=A)  ≈  P(Ŷ=1 | group=B)
    The positive-prediction rate is similar across groups.

  EQUAL OPPORTUNITY
    P(Ŷ=1 | Y=1, group=A) ≈ P(Ŷ=1 | Y=1, group=B)
    Among true positives, the catch rate (recall) is similar.

  EQUALIZED ODDS
    Equal opportunity AND equal P(Ŷ=1 | Y=0, group=*) (FPR parity).

Run: python 01_bias_fairness.py
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a synthetic dataset where a sensitive attribute correlates
# with the features (and through them, with the label).
# ───────────────────────────────────────────────────────────
n = 5000
X, y = make_classification(n_samples=n, n_features=8, weights=[0.6, 0.4], random_state=0)

# Sensitive attribute "group" — 0 or 1 — correlated with feature 0.
# This means the model will likely treat the groups differently.
group = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, group, test_size=0.3, random_state=0, stratify=y,
)


# ───────────────────────────────────────────────────────────
# Train a vanilla classifier
# ───────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"overall accuracy: {accuracy_score(y_test, y_pred):.3f}")


# ───────────────────────────────────────────────────────────
# Fairness MetricFrame — slice metrics by sensitive attribute
# ───────────────────────────────────────────────────────────
frame = MetricFrame(
    metrics={
        "selection_rate":      selection_rate,        # P(Ŷ=1)
        "true_positive_rate":  true_positive_rate,    # recall
        "accuracy":            accuracy_score,
    },
    y_true=y_test, y_pred=y_pred, sensitive_features=g_test,
)
print("\nper-group metrics:")
print(frame.by_group.round(3))


# ───────────────────────────────────────────────────────────
# Fairness summaries
# ───────────────────────────────────────────────────────────
print("\nfairness summaries:")
print(f"  demographic parity difference: "
      f"{demographic_parity_difference(y_test, y_pred, sensitive_features=g_test):.3f}")
print(f"  demographic parity ratio:      "
      f"{demographic_parity_ratio(y_test, y_pred, sensitive_features=g_test):.3f}")
print(f"  equalized odds difference:     "
      f"{equalized_odds_difference(y_test, y_pred, sensitive_features=g_test):.3f}")


# ───────────────────────────────────────────────────────────
# Mitigation — postprocessing equalized odds
# ───────────────────────────────────────────────────────────
from fairlearn.postprocessing import ThresholdOptimizer

mitigator = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
).fit(X_train, y_train, sensitive_features=g_train)
y_pred_mit = mitigator.predict(X_test, sensitive_features=g_test)

print("\nafter postprocessing for equalized odds:")
frame_mit = MetricFrame(
    metrics={"selection_rate": selection_rate,
             "true_positive_rate": true_positive_rate,
             "accuracy": accuracy_score},
    y_true=y_test, y_pred=y_pred_mit, sensitive_features=g_test,
)
print(frame_mit.by_group.round(3))
print(f"\noverall accuracy after mitigation: {accuracy_score(y_test, y_pred_mit):.3f}")
print(f"equalized odds difference:         "
      f"{equalized_odds_difference(y_test, y_pred_mit, sensitive_features=g_test):.3f}")


# ───────────────────────────────────────────────────────────
# Choosing the right metric
# ───────────────────────────────────────────────────────────
PICKING = """\
PICKING A FAIRNESS METRIC

DEMOGRAPHIC PARITY
  Use when the BASE RATE shouldn't predict who gets selected — e.g. ad
  serving for a job ad must not show only to one gender.

EQUAL OPPORTUNITY
  Use when MISSING TRUE POSITIVES has cost, and you want catch rates to
  be equal across groups — e.g. medical screening should catch sick
  patients equally regardless of subgroup.

EQUALIZED ODDS
  Strongest. Both TPR and FPR equal across groups. Use when both
  miss-detections and false alarms have asymmetric costs.

PROXIES — be careful: a feature like ZIP CODE often correlates with
RACE. Removing the SENSITIVE COLUMN doesn't make the model fair if the
proxies remain.

NOT EVERY DISPARITY IS UNFAIRNESS. Differences in base rates across
groups can be REAL (e.g. age × disease prevalence). The right approach
is domain expertise, not blind metric optimization.
"""
print(PICKING)


# ───────────────────────────────────────────────────────────
# Tools
# ───────────────────────────────────────────────────────────
TOOLS = """\
- fairlearn (Microsoft)         — what we used. sklearn-flavored API.
- AIF360 (IBM)                   — broader catalog of metrics + mitigators.
- What-If Tool (Google)          — interactive notebook tool.
- Aequitas                        — bias auditor with policy-friendly reports.
"""
print(TOOLS)
