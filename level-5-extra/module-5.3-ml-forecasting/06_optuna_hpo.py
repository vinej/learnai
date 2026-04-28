"""
06 — Hyperparameter optimization with Optuna.

Optuna runs Bayesian search (TPE by default) over the hyperparameter
space and supports pruning — stopping bad trials early based on
intermediate metrics. We use walk-forward CV inside each trial.

Run: python 06_optuna_hpo.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from _common import fetch_market

_spec = importlib.util.spec_from_file_location(
    "fe", pathlib.Path(__file__).parent / "01_feature_engineering.py"
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


def objective(trial: optuna.Trial, X, y) -> float:
    params = {
        "objective": "regression_l1",
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feat_frac", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bag_frac", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("l2", 1e-3, 10.0, log=True),
        "verbose": -1,
    }
    n_round = trial.suggest_int("n_round", 100, 600)

    tscv = TimeSeriesSplit(n_splits=4, gap=5, test_size=252)
    fold_maes = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        booster = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx]),
            num_boost_round=n_round,
        )
        pred = booster.predict(X.iloc[te_idx])
        mae = float(np.abs(pred - y.iloc[te_idx].values).mean())
        fold_maes.append(mae)
        # Pruning: report intermediate score; bad trials stop early.
        trial.report(np.mean(fold_maes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_maes))


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = fe.build_feature_matrix(px)
    feats = [c for c in df.columns if c not in ("close", "y")]
    X, y = df[feats], df["y"]

    study = optuna.create_study(direction="minimize",
                                 pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, X, y), n_trials=30, show_progress_bar=False)

    print("\nBest MAE :", round(study.best_value, 6))
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:<12} {v}")
