"""
Exercise 2 — Train, evaluate, register, promote

Walk through the production-style flow:

  1. Train three candidate models, log each with metrics.
  2. Pick the best by held-out ROC-AUC.
  3. Register that one as version N of "ex2-classifier".
  4. Set the @champion alias on it.
  5. Load `models:/ex2-classifier@champion` and confirm it predicts.

Run: python exercises/02_promote_best_model.py
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REGISTERED_NAME = "ex2-classifier"
mlflow.set_experiment("ex2_promote")


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)


def train_run(name: str, model):
    """Train + log a single candidate. Returns (run_id, test_score)."""
    with mlflow.start_run(run_name=name) as run:
        for k, v in model.get_params().items():
            mlflow.log_param(k, v)
        model.fit(X_train, y_train)
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_roc_auc", score)
        info = mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id, score, info.model_uri


def main():
    candidates = [
        ("logreg",  LogisticRegression(C=1.0, max_iter=2000)),
        ("rf",      RandomForestClassifier(n_estimators=200, random_state=0)),
        ("gbm",     GradientBoostingClassifier(n_estimators=200, random_state=0)),
    ]

    results = []
    for name, model in candidates:
        run_id, score, model_uri = train_run(name, model)
        print(f"  {name:<8}  run={run_id[:8]}  test_roc_auc={score:.4f}")
        results.append((name, run_id, score, model_uri))

    # Pick the best
    best_name, best_run_id, best_score, best_uri = max(results, key=lambda r: r[2])
    print(f"\nbest: {best_name}  ({best_score:.4f})")

    # Register that one specifically
    client = MlflowClient()
    registered = mlflow.register_model(model_uri=best_uri, name=REGISTERED_NAME)
    version = int(registered.version)
    print(f"registered as {REGISTERED_NAME} v{version}")

    # Promote to @champion
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version)
    print(f"@champion alias → v{version}")

    # Load it back as a downstream service would
    champion = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    pred_score = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])
    print(f"\nchampion reloaded; ROC-AUC: {pred_score:.4f}")

    assert abs(pred_score - best_score) < 1e-9, \
        "reloaded model should match the run's metric exactly"
    print("registry round-trip works ✅")


if __name__ == "__main__":
    main()
