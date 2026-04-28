"""
02 — Explainability

A model that "just works" but no one can explain has TWO problems:
  • Hard to debug failure cases.
  • Often impossible to deploy in regulated domains.

Two scopes:
  GLOBAL    — average behavior across the dataset. "Which features matter most?"
  LOCAL     — explanation for ONE prediction. "Why did the model say this for THIS user?"

Methods:
  Permutation importance      — model-agnostic. Shuffle a feature; how much
                                does accuracy drop?  GLOBAL.
  SHAP                         — Shapley values. Rigorous; both LOCAL & GLOBAL.
  LIME                         — fits a simple model around one prediction. LOCAL.
  Integrated gradients         — for DL; attributes prediction to input features.
  For LLMs: chain-of-thought   — ask the model to justify; treat as ATTESTATION,
                                not explanation. Audit-trail, not ground truth.

This file demonstrates permutation importance with sklearn (no extra deps),
plus a SHAP snippet you can run if you `pip install shap`.

Run: python 02_explainability.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ───────────────────────────────────────────────────────────
# Train a model
# ───────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = GradientBoostingRegressor(random_state=0, n_estimators=200).fit(X_train, y_train)
print(f"R² on test: {model.score(X_test, y_test):.3f}")


# ───────────────────────────────────────────────────────────
# Global: permutation importance
# ───────────────────────────────────────────────────────────
result = permutation_importance(
    model, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1,
)
imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
print("\npermutation importance:")
print(imp.round(3))


# ───────────────────────────────────────────────────────────
# Local: a single prediction's tree-based attribution
# ───────────────────────────────────────────────────────────
# Without SHAP installed, we use a simple approximation:
# show the SAMPLE feature values vs the dataset means.
sample_idx = 0
sample = X_test[sample_idx]
pred = model.predict([sample])[0]
print(f"\nfor test sample {sample_idx}: predicted = {pred:.3f}")

means = X_train.mean(axis=0)
deltas = sample - means
print("\nfeature   sample  mean    delta  importance")
for i, name in enumerate(feature_names):
    print(f"  {name:<10}  {sample[i]:>6.2f}  {means[i]:>6.2f}  {deltas[i]:>+6.2f}    {result.importances_mean[i]:.3f}")


# ───────────────────────────────────────────────────────────
# SHAP — much sharper local & global explanations
# ───────────────────────────────────────────────────────────
SHAP_SNIPPET = """\
# pip install shap
import shap

# TreeExplainer is fast and exact for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global summary
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Per-prediction force plot
shap.plots.waterfall(
    shap.Explanation(values=shap_values[0],
                     base_values=explainer.expected_value,
                     data=X_test[0],
                     feature_names=feature_names)
)
"""
print(f"\n=== SHAP snippet (runs after `pip install shap`) ===")
print(SHAP_SNIPPET)


# ───────────────────────────────────────────────────────────
# Explainability for LLMs
# ───────────────────────────────────────────────────────────
LLM_NOTES = """\
LLM "EXPLANATIONS" — how to think about them

CHAIN OF THOUGHT
  Asking the model to "think step by step" produces a TRACE that LOOKS
  like reasoning. Treat it as an ATTESTATION the model gives — useful
  for transparency and debugging, but NOT a ground truth of why it
  predicted what it did. Models can produce post-hoc rationalizations.

ATTENTION MAPS / ATTRIBUTION
  Tools like attention rollout or integrated gradients assign credit to
  input tokens. Useful for research; rarely communicates well to end users.

PROVENANCE / CITATIONS
  Often more useful than internal explanations: "this answer cites doc X
  paragraph Y" tells the user what to verify.

FOR REGULATED USE CASES
  - Combine: a deterministic model component (e.g. a credit-scoring layer)
    that IS explainable, plus an LLM as a "wrapper" for explanation.
  - Always log inputs / outputs / model versions for audit replay.
  - Don't claim more than you can prove. "The model gave X based on Y"
    must be substantiated with the actual mechanism.
"""
print(LLM_NOTES)


# ───────────────────────────────────────────────────────────
# Communicating uncertainty
# ───────────────────────────────────────────────────────────
UNCERTAINTY = """\
A great explanation is BOUNDED — it tells the reader what's certain,
what's a guess, and what's missing.

Tactics:
  - Predict a CALIBRATED PROBABILITY (or interval), not a point.
  - Show feature values that drove the prediction AND outliers worth
    inspecting.
  - State the model's TRAINING DATA RANGE; warn on out-of-distribution
    inputs.
  - For LLMs: surface "I don't know" prominently when retrieval misses.
"""
print(UNCERTAINTY)
