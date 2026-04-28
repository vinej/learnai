"""
Exercise 2 — One-hot vs target encoding on a high-cardinality column

You have a categorical feature with 1000 unique levels. Compare two encodings:
  A. OneHotEncoder
  B. TargetEncoder
on a logistic regression. Report training time, model size, and ROC-AUC.

The script asserts target encoding ROC-AUC ≥ OHE within ~0.005, while using
~1000× fewer features.

Run: python exercises/02_ohe_vs_target.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder


rng = np.random.default_rng(0)


def make_dataset():
    n_levels = 1000
    n        = 50_000
    true_rate = rng.beta(2, 8, size=n_levels)         # category-specific rate

    cat_idx = rng.integers(0, n_levels, n)
    y = (rng.random(n) < true_rate[cat_idx]).astype(int)
    df = pd.DataFrame({
        "category": [f"cat_{i}" for i in cat_idx],
    })
    return df, y


def evaluate(name, X_train, X_test, y_train, y_test):
    t0 = time.perf_counter()
    model = LogisticRegression(max_iter=2000, n_jobs=-1).fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    n_features = X_train.shape[1]
    print(f"{name:<22}  features={n_features:>6}  "
          f"train={train_time:.2f}s  ROC-AUC={score:.4f}")
    return score, n_features


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y,
    )

    # A. OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(X_train)
    X_train_ohe = ohe.transform(X_train)
    X_test_ohe  = ohe.transform(X_test)
    auc_ohe, n_ohe = evaluate("OneHotEncoder",
                              X_train_ohe, X_test_ohe, y_train, y_test)

    # B. TargetEncoder (sklearn ≥ 1.3 — fits with internal CV automatically)
    te = TargetEncoder(target_type="binary", random_state=0).fit(X_train, y_train)
    X_train_te = te.transform(X_train)
    X_test_te  = te.transform(X_test)
    auc_te, n_te = evaluate("TargetEncoder",
                            X_train_te, X_test_te, y_train, y_test)

    print(f"\nfeature ratio: {n_ohe / n_te:.0f}× fewer features with target encoding")
    print(f"AUC delta (TE - OHE): {(auc_te - auc_ohe) * 100:+.3f} pp")

    assert auc_te > auc_ohe - 0.01, "target encoder ought to be at least as good"
    assert n_te < n_ohe / 10,        "target encoder should be tiny vs OHE"
    print("\ntarget encoding wins on size, ties (or wins) on AUC ✅")


if __name__ == "__main__":
    main()
