"""
Exercise 1 — Predict California house prices end-to-end

Compare three models on RMSE and report which one wins:

  • Linear Ridge regression
  • Random Forest
  • XGBoost

Each model goes inside a Pipeline that handles scaling (where it helps).
The script reports test-set RMSE and asserts that XGBoost beats Ridge.

Run: python exercises/01_california_housing.py
"""

import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def report(name, model, X_train, X_test, y_train, y_test):
    t = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t
    pred = model.predict(X_test)
    print(f"{name:<22}  RMSE={rmse(y_test, pred):.3f}  "
          f"MAE={mean_absolute_error(y_test, pred):.3f}  "
          f"R²={r2_score(y_test, pred):.3f}  "
          f"trained in {train_time:.2f}s")
    return rmse(y_test, pred)


def main():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"train size: {X_train.shape}, test size: {X_test.shape}\n")

    rmses = {}

    # 1. Ridge regression baseline
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    rmses["ridge"] = report("Ridge (linear)", ridge, X_train, X_test, y_train, y_test)

    # 2. Random Forest — no scaling needed
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    rmses["rf"] = report("Random Forest", rf, X_train, X_test, y_train, y_test)

    # 3. XGBoost
    xgbm = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=20,
    )
    t = time.perf_counter()
    xgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = xgbm.predict(X_test)
    print(f"{'XGBoost':<22}  RMSE={rmse(y_test, pred):.3f}  "
          f"MAE={mean_absolute_error(y_test, pred):.3f}  "
          f"R²={r2_score(y_test, pred):.3f}  "
          f"trained in {time.perf_counter() - t:.2f}s   "
          f"best iter={xgbm.best_iteration}")
    rmses["xgb"] = rmse(y_test, pred)

    print("\nranking (lower RMSE = better):")
    for name, score in sorted(rmses.items(), key=lambda x: x[1]):
        print(f"  {name:<6} {score:.3f}")

    assert rmses["xgb"] < rmses["ridge"], "XGBoost should beat Ridge here"
    print("\nXGBoost beats Ridge ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add LightGBM to the comparison.
# • Use 5-fold CV instead of a single split — does the ranking still hold?
# • Plot residuals (y_test - y_pred) for each model.
# ─────────────────────────────────────────────────────────────────────────────
