"""
05 — Target / mean encoding

Replace each category with the AVERAGE TARGET VALUE seen for that category in
the training set:

    "Paris" → 0.31   ("Paris" customers had a 31% conversion rate")
    "Tokyo" → 0.18

For high-cardinality features (cities, products, zip codes), this beats one-hot
on linear models AND tree models — IF you do it without leaking.

LEAKAGE: if you compute the mean using the row you're encoding, the model
already knows the answer. This makes train scores look amazing and test
scores collapse.

THE FIX: use cross-validation when computing the encoding (sklearn's
TargetEncoder does this automatically), and SMOOTH toward the global mean
for rare levels.

Run: python 05_target_encoding.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a high-cardinality categorical with a real signal
# ───────────────────────────────────────────────────────────
# 1000 fake "cities", each with its own true conversion rate
n_cities = 1000
true_rates = rng.beta(2, 8, size=n_cities)        # mostly low, some high

n = 50_000
city_idx = rng.integers(0, n_cities, size=n)
y = (rng.random(n) < true_rates[city_idx]).astype(int)

df = pd.DataFrame({
    "city": [f"city_{i}" for i in city_idx],
    "y": y,
})
print(f"{n_cities} unique cities, {n} samples, "
      f"overall positive rate {y.mean():.3f}")


# ───────────────────────────────────────────────────────────
# Split BEFORE doing any encoding
# ───────────────────────────────────────────────────────────
X = df[["city"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, df["y"].values, test_size=0.25, random_state=0, stratify=df["y"],
)


# ───────────────────────────────────────────────────────────
# Approach A — One-hot encoding (1000 columns)
# ───────────────────────────────────────────────────────────
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe  = ohe.transform(X_test)
print(f"\nOHE matrix shape: {X_train_ohe.shape}  (1 col per unique city)")

ohe_model = LogisticRegression(max_iter=1000).fit(X_train_ohe, y_train)
auc_ohe = roc_auc_score(y_test, ohe_model.predict_proba(X_test_ohe)[:, 1])
print(f"  test ROC-AUC: {auc_ohe:.4f}")


# ───────────────────────────────────────────────────────────
# Approach B — TargetEncoder (single column!)
# ───────────────────────────────────────────────────────────
# sklearn's TargetEncoder fits the encoding using internal cross-validation
# on the TRAINING data only. It also smooths rare categories toward the
# global mean (cv default 5, smooth='auto').
te = TargetEncoder(target_type="binary", random_state=0)
X_train_te = te.fit_transform(X_train, y_train)
X_test_te  = te.transform(X_test)
print(f"\ntarget-encoded matrix shape: {X_train_te.shape}  (just 1 column)")

te_model = LogisticRegression(max_iter=1000).fit(X_train_te, y_train)
auc_te = roc_auc_score(y_test, te_model.predict_proba(X_test_te)[:, 1])
print(f"  test ROC-AUC: {auc_te:.4f}")


# ───────────────────────────────────────────────────────────
# Approach C — Naive (LEAKY) target encoding — for contrast
# ───────────────────────────────────────────────────────────
# Compute the mean target per category on the FULL dataset and apply it.
# Looks great in train, but:
#   1. It uses the row itself in computing its own encoding (data leakage).
#   2. It may overfit to rare-category accidents.
naive_means = df.groupby("city")["y"].mean()
df["city_naive_te"] = df["city"].map(naive_means)
naive_train = df.loc[X_train.index, "city_naive_te"].values.reshape(-1, 1)
naive_test  = df.loc[X_test.index,  "city_naive_te"].values.reshape(-1, 1)

naive_model = LogisticRegression(max_iter=1000).fit(naive_train, y_train)
auc_naive = roc_auc_score(y_test, naive_model.predict_proba(naive_test)[:, 1])
print(f"\nleaky 'naive' target encoding ROC-AUC: {auc_naive:.4f}")
print("→ this is the score that LIES; it'd be lower in production.")


# ───────────────────────────────────────────────────────────
# Recap
# ───────────────────────────────────────────────────────────
# • Use sklearn's TargetEncoder (≥ 1.3) — it handles CV and smoothing for you.
# • Always SPLIT BEFORE encoding. Always wrap in a Pipeline so transforms
#   are re-fit per fold during CV.
# • For super-high-cardinality + linear models, target encoding is a huge win
#   over one-hot (1000 cols → 1 col, similar or better accuracy).
