"""
Exercise 1 — Track an Optuna sweep with MLflow

Run an Optuna study and log every trial as a child run under one parent run.
Then query the runs to find the best one.

The script asserts:
  • The number of MLflow runs in the sweep parent equals the number of trials.
  • The "best" run reported by Optuna matches the best run in MLflow.

Run: python exercises/01_track_a_sweep.py
"""

import mlflow
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_experiment("ex1_optuna_sweep")


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)


def train_and_log(params: dict, parent_run_id: str) -> float:
    """Train one configuration as a CHILD run under the sweep parent."""
    with mlflow.start_run(nested=True, parent_run_id=parent_run_id):
        mlflow.log_params(params)
        model = GradientBoostingClassifier(**params, random_state=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        cv_score = cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="roc_auc").mean()
        mlflow.log_metric("cv_roc_auc", cv_score)

        # Also fit final and log test score
        model.fit(X_train, y_train)
        test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_roc_auc", test_score)
        return cv_score


def main():
    n_trials = 12

    with mlflow.start_run(run_name="optuna-sweep") as parent:
        mlflow.set_tag("sweep", "optuna")
        mlflow.log_param("n_trials", n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
                "max_depth":     trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            }
            return train_and_log(params, parent.info.run_id)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        mlflow.log_metric("best_cv_roc_auc", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    print(f"sweep best CV ROC-AUC: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    # ── Verify with MLflow ──
    children = mlflow.search_runs(
        experiment_names=["ex1_optuna_sweep"],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    print(f"\n{len(children)} child runs logged in MLflow")
    best_in_mlflow = children["metrics.cv_roc_auc"].max()

    assert len(children) == n_trials, "child run count should match trial count"
    assert abs(best_in_mlflow - study.best_value) < 1e-6, \
        "best in MLflow should match Optuna's best"
    print("\nsweep tracked correctly ✅")


if __name__ == "__main__":
    main()
