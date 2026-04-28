"""
03 — XGBoost and LightGBM

The standard tabular workhorses outside of sklearn:

  XGBoost     — eXtreme Gradient Boosting. Very strong, well-tuned defaults.
  LightGBM    — Histogram-based, much faster; uses leaf-wise growth.
  CatBoost    — Native categorical support; great for high-cardinality cats
                (not installed here — pip install catboost if you want it).

All three follow the sklearn-compatible API: .fit(), .predict(), .predict_proba().
They also accept pandas DataFrames directly.

Run: python 03_xgboost_lightgbm.py
"""

import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ───────────────────────────────────────────────────────────
# XGBoost
# ───────────────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,            # use 80% of rows per tree (regularization)
    colsample_bytree=0.8,     # use 80% of features per tree
    random_state=0,
    n_jobs=-1,
    tree_method="hist",       # fast histogram method (default in 2.x)
    early_stopping_rounds=20, # stop if val loss doesn't improve for 20 rounds
)

t = time.perf_counter()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
xgb_time = time.perf_counter() - t

xgb_rmse = mean_squared_error(y_test, xgb_model.predict(X_test)) ** 0.5
print(f"XGBoost:   RMSE={xgb_rmse:.3f}   trained in {xgb_time:.2f}s   "
      f"best iter={xgb_model.best_iteration}")


# ───────────────────────────────────────────────────────────
# LightGBM
# ───────────────────────────────────────────────────────────
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,            # the LightGBM "complexity" knob (vs max_depth)
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1,
    verbose=-1,
)

t = time.perf_counter()
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
)
lgb_time = time.perf_counter() - t

lgb_rmse = mean_squared_error(y_test, lgb_model.predict(X_test)) ** 0.5
print(f"LightGBM:  RMSE={lgb_rmse:.3f}   trained in {lgb_time:.2f}s   "
      f"best iter={lgb_model.best_iteration_}")


# ───────────────────────────────────────────────────────────
# Feature importance
# ───────────────────────────────────────────────────────────
import pandas as pd
imp = pd.DataFrame({
    "feature": fetch_california_housing().feature_names,
    "xgb":     xgb_model.feature_importances_,
    "lgb":     lgb_model.feature_importances_,
})
print("\nfeature importance:")
print(imp.sort_values("xgb", ascending=False).reset_index(drop=True))


# ───────────────────────────────────────────────────────────
# Tuning checklist
# ───────────────────────────────────────────────────────────
# Big four hyperparameters (XGBoost / LightGBM):
#   n_estimators       — usually leave large, let early stopping trim it
#   learning_rate      — start at 0.05; smaller + more rounds = better
#   max_depth / num_leaves
#                      — depth 5-8, num_leaves 15-127 typical
#   subsample / colsample_bytree
#                      — 0.7-0.9 for stochastic regularization
# See module 2.2.08_hyperparameter_search.py for tuning with Optuna.


# ───────────────────────────────────────────────────────────
# Why these usually beat plain sklearn GBM
# ───────────────────────────────────────────────────────────
# • Histogram binning of features → 10-50× faster training.
# • Native handling of missing values (XGBoost & LightGBM both).
# • LightGBM's leaf-wise growth often gives better accuracy faster.
# • GPU support, distributed training, well-tuned defaults.
