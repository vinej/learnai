"""
07 — Pipelines and ColumnTransformer

A Pipeline chains preprocessing + modeling into a single estimator.

WHY pipelines (besides ergonomics):
  • LEAKAGE PREVENTION — preprocessors are fit only on training data,
    automatically, even inside cross-validation.
  • REPRODUCIBILITY    — one object captures the entire transform chain.
  • DEPLOYMENT         — save the pipeline, load it, predict. No "did I
    apply the same scaler?" anxiety.

ColumnTransformer applies different preprocessors to different columns,
which is essential for mixed-type tabular data.

Run: python 07_pipelines.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ───────────────────────────────────────────────────────────
# Build a synthetic mixed-type dataset
# ───────────────────────────────────────────────────────────
# We don't want to depend on a network fetch, so build one ourselves.
rng = np.random.default_rng(0)
n = 800

df = pd.DataFrame({
    "age":          rng.integers(18, 80, n),
    "income":       rng.normal(50_000, 15_000, n),
    "tenure_years": rng.gamma(2.0, 2.0, n),
    "city":         rng.choice(["Paris", "London", "Berlin", "Madrid"], n),
    "plan":         rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
})
# Drop some values to make imputation matter
mask = rng.random(n) < 0.1
df.loc[mask, "income"] = np.nan
mask = rng.random(n) < 0.05
df.loc[mask, "city"] = None

# Synthetic target: "churn" depends on tenure + plan + a bit of randomness
logits = (
    -1.0 * (df["tenure_years"] / 5)
    + 0.5 * (df["plan"] == "basic")
    - 0.2 * (df["plan"] == "enterprise")
    + rng.normal(0, 0.5, n)
)
y = (logits > -0.3).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25,
                                                    random_state=0, stratify=y)


# ───────────────────────────────────────────────────────────
# A handcrafted ColumnTransformer
# ───────────────────────────────────────────────────────────
# Numeric columns:   impute (median) → scale
# Categorical cols:  impute (most_frequent) → one-hot encode
numeric_features = ["age", "income", "tenure_years"]
categorical_features = ["city", "plan"]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# Stick the preprocessor + a model together
pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)
print(f"logistic pipeline accuracy: {accuracy_score(y_test, pipe.predict(X_test)):.3f}")


# ───────────────────────────────────────────────────────────
# Cross-validate the WHOLE pipeline
# ───────────────────────────────────────────────────────────
# Each fold re-fits the imputer/scaler/encoder on the training portion only.
# That's leakage prevention without you having to think about it.
scores = cross_val_score(pipe, df, y, cv=5)
print(f"5-fold CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")


# ───────────────────────────────────────────────────────────
# make_column_selector — automatic dtype-based selection
# ───────────────────────────────────────────────────────────
# Saves typing for big tables. Selects all numeric or all object/category cols.
auto_preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,    make_column_selector(dtype_include="number")),
    ("cat", categorical_pipeline, make_column_selector(dtype_include="object")),
])

auto_pipe = Pipeline([("prep", auto_preprocessor),
                      ("model", HistGradientBoostingClassifier(random_state=0))])
auto_pipe.fit(X_train, y_train)
print(f"\nauto-selected GBM accuracy: {accuracy_score(y_test, auto_pipe.predict(X_test)):.3f}")


# ───────────────────────────────────────────────────────────
# Inspecting / accessing components
# ───────────────────────────────────────────────────────────
# Pipelines are dict-like: pipe.named_steps['model']  or  pipe[-1]
print("\npipeline steps:", [name for name, _ in pipe.steps])
print("OHE feature names:")
print(pipe.named_steps["prep"].named_transformers_["cat"]
      .named_steps["onehot"].get_feature_names_out(categorical_features))


# ───────────────────────────────────────────────────────────
# Things to know
# ───────────────────────────────────────────────────────────
# • OneHotEncoder(handle_unknown="ignore"): test set may have a category
#   the train set never saw. Without this, predict() will crash.
# • Tree-based models (RF / GBM / XGBoost) DON'T need scaling. Drop the
#   scaler — it just slows things down and complicates inspection.
# • For categorical features with high cardinality (cities, products), use
#   a target/mean encoder instead of one-hot. See module 2.3.
