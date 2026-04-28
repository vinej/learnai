"""
Exercise 2 — Full preprocessing pipeline

Build a single Pipeline that:
  1. Imputes missing values
  2. Scales numeric features
  3. One-hot encodes a categorical feature
  4. Trains a classifier

Then evaluate with cross-validation. The script asserts CV mean accuracy
> 0.9 — easy to hit if you wire up the pipeline correctly.

Run: python exercises/02_wine_pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_dataset() -> pd.DataFrame:
    """Load wine and inject realism: a categorical column + missing values."""
    rng = np.random.default_rng(0)
    data = load_wine(as_frame=True)
    df = data.frame.copy()              # numeric features + 'target'
    df = df.rename(columns={"target": "_y"})

    # Add a synthetic categorical column. NOT derived from the target
    # (that would be leakage).
    df["estate"] = rng.choice(["A", "B", "C", "D"], size=len(df))

    # Inject missing values into a few numeric columns
    for col in ["alcohol", "malic_acid", "color_intensity"]:
        mask = rng.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan

    # Inject a few missing categoricals too
    mask = rng.random(len(df)) < 0.05
    df.loc[mask, "estate"] = None

    return df


def build_pipeline() -> Pipeline:
    """A canonical 4-step pipeline. Customize the model if you like."""
    numeric_features = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "color_intensity", "hue",
        "od280/od315_of_diluted_wines", "proline",
    ]
    categorical_features = ["estate"]

    # ▼▼▼ Your code: build numeric_pipe, categorical_pipe, preprocessor, model ▼▼▼
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(max_iter=2000, multi_class="multinomial")),
    ])
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


def main():
    df = make_dataset()
    y = df["_y"].values
    X = df.drop(columns="_y")

    print(f"shape: {X.shape}")
    print(f"missing per column:\n{X.isna().sum()}")

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")
    print(f"per-fold: {scores.round(3).tolist()}")

    assert scores.mean() > 0.9, "your pipeline should hit > 0.9 mean CV accuracy"
    print("pipeline meets the bar ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Swap the model for HistGradientBoostingClassifier — does it help?
# • Replace OneHotEncoder with TargetEncoder (sklearn ≥ 1.3) and compare.
# • Add a feature-selection step (SelectKBest, RFE) before the model.
# ─────────────────────────────────────────────────────────────────────────────
