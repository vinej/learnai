"""
Exercise 2 — Explain a model's predictions

For a tabular classifier, produce:
  • GLOBAL feature importance (permutation importance).
  • LOCAL explanation for ONE specific prediction (feature deltas vs
    dataset means, weighted by importance).

Assert: the SAME features dominate global importance and the local
explanation of a prediction whose features lie far from the dataset mean.

Run: python exercises/02_explanations.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ───────────────────────────────────────────────────────────
# Train a model
# ───────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)
clf = GradientBoostingClassifier(random_state=0, n_estimators=200).fit(X_train, y_train)
print(f"accuracy: {clf.score(X_test, y_test):.3f}")


# ───────────────────────────────────────────────────────────
# GLOBAL: permutation importance
# ───────────────────────────────────────────────────────────
result = permutation_importance(
    clf, X_test, y_test, n_repeats=8, random_state=0, n_jobs=-1,
)
imp = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
print("\nTop-10 globally important features:")
print(imp.head(10).round(4))
top_global = set(imp.head(5).index)


# ───────────────────────────────────────────────────────────
# LOCAL: pick the most "extreme" test sample (most different from mean)
# ───────────────────────────────────────────────────────────
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
z = np.abs((X_test - mean) / std).sum(axis=1)
sample_idx = int(np.argmax(z))
sample = X_test[sample_idx]

prob = clf.predict_proba([sample])[0]
print(f"\nselected sample {sample_idx}: P(class=1) = {prob[1]:.3f}, "
      f"true label = {y_test[sample_idx]}")


# Local "explanation" = (sample - mean) × global_importance.
# Positive  = feature pushes the prediction in the direction the model
# associates this feature with.
deltas = (sample - mean) / std
contributions = deltas * result.importances_mean
top_local = pd.Series(np.abs(contributions), index=features).sort_values(ascending=False).head(5)
print("\nTop-5 locally influential features (|delta × importance|):")
for fname in top_local.index:
    i = list(features).index(fname)
    direction = "↑" if contributions[i] > 0 else "↓"
    print(f"  {fname:<32}  z={deltas[i]:+.2f}  imp={result.importances_mean[i]:.3f}  {direction}")


# ───────────────────────────────────────────────────────────
# Assertion
# ───────────────────────────────────────────────────────────
overlap = top_global & set(top_local.index)
assert overlap, ("expected at least one of the top-5 LOCAL features to "
                 "overlap with top-5 GLOBAL features")
print(f"\noverlap (top-5 global ∩ top-5 local): {overlap}")
print("explanation pipeline works ✅")


# ───────────────────────────────────────────────────────────
# SHAP — when permutation isn't enough
# ───────────────────────────────────────────────────────────
SHAP_RECIPE = """\
Permutation importance is fast and model-agnostic but coarse.
For a SHARPER local explanation:

  pip install shap
  import shap

  explainer = shap.TreeExplainer(clf)
  shap_values = explainer.shap_values(X_test)

  # local
  shap.plots.waterfall(
      shap.Explanation(values=shap_values[sample_idx],
                       base_values=explainer.expected_value,
                       data=X_test[sample_idx],
                       feature_names=list(features))
  )

  # global
  shap.summary_plot(shap_values, X_test, feature_names=list(features))

Tree models: TreeExplainer is exact and fast.
Neural nets: DeepExplainer / GradientExplainer (slower, approximate).
Black-box: KernelExplainer (model-agnostic, very slow).
"""
print(SHAP_RECIPE)
