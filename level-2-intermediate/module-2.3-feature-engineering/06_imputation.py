"""
06 — Imputing missing values

When you can't drop the missing rows, you have to fill them. Choices:

  SimpleImputer        — mean / median / most_frequent / constant.
                         Fast, the default first move.
  KNNImputer           — fill from the K nearest neighbors. Captures
                         feature relationships, slower.
  IterativeImputer     — model each missing column with regression on the
                         others; iterate. Most powerful, slowest.

Plus a bonus trick: ADD A "WAS MISSING" FLAG. Sometimes the fact that a
value was missing IS the most predictive feature.

Run: python 06_imputation.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required side-effect)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Data with planted missingness
# ───────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X = X.copy()

# Knock out 10% of values at random
mask = rng.random(X.shape) < 0.10
X[mask] = np.nan
print(f"missing rate: {np.isnan(X).mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)


# ───────────────────────────────────────────────────────────
# Compare imputers
# ───────────────────────────────────────────────────────────
imputers = {
    "SimpleImputer (mean)":   SimpleImputer(strategy="mean"),
    "SimpleImputer (median)": SimpleImputer(strategy="median"),
    "KNNImputer":             KNNImputer(n_neighbors=5),
    "IterativeImputer":       IterativeImputer(max_iter=10, random_state=0),
}

print(f"\n{'imputer':<26}  test accuracy")
for name, imp in imputers.items():
    pipe = make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_train, y_train)
    score = accuracy_score(y_test, pipe.predict(X_test))
    print(f"{name:<26}  {score:.3f}")


# ───────────────────────────────────────────────────────────
# Missingness as a feature
# ───────────────────────────────────────────────────────────
# add_indicator=True adds binary "was missing" columns alongside the imputed
# values. Try this whenever missingness is non-random (e.g. survey skips,
# system errors, opt-outs). Often a free accuracy bump.
imp_with_flag = SimpleImputer(strategy="median", add_indicator=True)
pipe_flagged = make_pipeline(imp_with_flag, StandardScaler(),
                             LogisticRegression(max_iter=2000))
pipe_flagged.fit(X_train, y_train)
score = accuracy_score(y_test, pipe_flagged.predict(X_test))
print(f"\nmedian + missingness-flags: {score:.3f}")


# ───────────────────────────────────────────────────────────
# Group-wise imputation (manual)
# ───────────────────────────────────────────────────────────
# Sometimes "fill with the global mean" is too crude. Filling with the mean
# of a meaningful group (city, department, season) preserves more signal.
df = pd.DataFrame({
    "city":   ["Paris", "Paris", "London", "London", "Paris"],
    "income": [50_000, np.nan, 70_000, np.nan, 55_000],
})
df["income"] = df.groupby("city")["income"].transform(
    lambda s: s.fillna(s.median())
)
print("\ngroup-wise imputation:")
print(df)


# ───────────────────────────────────────────────────────────
# Picking what to do
# ───────────────────────────────────────────────────────────
# • Quick win                        → SimpleImputer(median) + add_indicator.
# • Features clearly correlated      → KNNImputer or IterativeImputer.
# • Big production data, speed matters → SimpleImputer (everything else is slow).
# • Tree models (XGBoost, LightGBM, HistGradientBoosting) handle NaN natively
#   — sometimes leaving them alone beats any imputation.


# ───────────────────────────────────────────────────────────
# Common mistakes
# ───────────────────────────────────────────────────────────
# 1. Imputing on the WHOLE dataset before splitting (leakage). Use a Pipeline.
# 2. Filling categorical NaNs with the global mode without checking — the
#    "missingness" might itself be a category worth keeping.
# 3. Leaving 0 in for missing → in many models, 0 is a real, meaningful
#    value. Always represent missing with NaN and impute deliberately.
