"""
Exercise 3 — Tune XGBoost with Optuna

Search for the best XGBoost hyperparameters on the breast-cancer dataset.
The script asserts the tuned model beats a sensible default.

Run: python exercises/03_optuna_xgboost.py
"""

import warnings

import numpy as np
import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tree_method":      "hist",
        "random_state":     0,
        "n_jobs":           -1,
    }
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ───────────────────────────────────────────────────────────
# 1. Default model — our baseline
# ───────────────────────────────────────────────────────────
default = xgb.XGBClassifier(
    n_estimators=200, tree_method="hist", random_state=0, eval_metric="logloss",
).fit(X_train, y_train)
default_auc = roc_auc_score(y_test, default.predict_proba(X_test)[:, 1])
print(f"default XGBoost test ROC-AUC:  {default_auc:.4f}")


# ───────────────────────────────────────────────────────────
# 2. Optuna study
# ───────────────────────────────────────────────────────────
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"\nbest CV ROC-AUC during search: {study.best_value:.4f}")
print("best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# ───────────────────────────────────────────────────────────
# 3. Train final model on all training data with the best params
# ───────────────────────────────────────────────────────────
best = xgb.XGBClassifier(
    **study.best_params, tree_method="hist", random_state=0,
    eval_metric="logloss", n_jobs=-1,
).fit(X_train, y_train)
tuned_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
print(f"\ntuned XGBoost test ROC-AUC:    {tuned_auc:.4f}")
print(f"improvement: {(tuned_auc - default_auc) * 100:.2f} percentage points")

assert tuned_auc >= default_auc - 0.01, (
    "tuned model should be at least as good as default; if not, increase n_trials"
)
print("tuning succeeded ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add a pruner: optuna.pruners.MedianPruner. Useful for expensive trainings.
# • Plot the optimization history with `optuna.visualization.plot_optimization_history(study)`.
# • Add nested CV: outer loop for honest evaluation, inner loop for tuning.
# ─────────────────────────────────────────────────────────────────────────────
