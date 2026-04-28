"""
05 — Model registry: champions, challengers, and stable URIs

A run has an opaque ID — fine for experiments, painful for production. The
MODEL REGISTRY gives each model a STABLE NAME with versions and aliases:

    models:/loan-classifier@champion       ← what production loads
    models:/loan-classifier@challenger     ← what you're shadow-testing
    models:/loan-classifier/3              ← specific version

Workflow:
  1. Train a model in a run, log it.
  2. Register it in the registry → it gets version N.
  3. Set an alias ("champion", "challenger", "staging") to a version.
  4. Production loads `models:/<name>@<alias>`. Promoting a new model is
     just moving an alias.

Note: the OLDER `stage` API ("Production", "Staging") is deprecated in
modern MLflow. Use ALIASES instead.

Run: python 05_model_registry.py
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REGISTERED_NAME = "breast-cancer-classifier"
mlflow.set_experiment("05_registry")


# ───────────────────────────────────────────────────────────
# Train two candidates and log each as a registered model
# ───────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)


def train_and_register(name: str, model) -> int:
    """Train, log to a run, register the run's model artifact. Return the new version."""
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("roc_auc", score)
        # log_model(..., registered_model_name=...) auto-registers a new version
        info = mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=REGISTERED_NAME,
        )
    # `info.registered_model_version` is the version number of this registration.
    return int(info.registered_model_version)


v1 = train_and_register("rf-100", RandomForestClassifier(n_estimators=100, random_state=0))
v2 = train_and_register("gbm-200", GradientBoostingClassifier(n_estimators=200, random_state=0))
print(f"registered '{REGISTERED_NAME}' versions: v{v1} and v{v2}")


# ───────────────────────────────────────────────────────────
# Set an alias for production
# ───────────────────────────────────────────────────────────
client = MlflowClient()

# Compare the two versions' metrics, pick the best one as champion.
def get_metric(version: int, key: str) -> float:
    mv = client.get_model_version(REGISTERED_NAME, str(version))
    run = client.get_run(mv.run_id)
    return float(run.data.metrics[key])


scores = {v: get_metric(v, "roc_auc") for v in [v1, v2]}
best = max(scores, key=scores.get)
print(f"\nscores: {scores}")
print(f"promoting v{best} to @champion")

client.set_registered_model_alias(REGISTERED_NAME, "champion", best)
client.set_registered_model_alias(REGISTERED_NAME, "challenger",
                                  v1 if best == v2 else v2)


# ───────────────────────────────────────────────────────────
# Production-style load
# ───────────────────────────────────────────────────────────
# A serving job uses the alias, not a version number:
#
#   model_uri = f"models:/{REGISTERED_NAME}@champion"
#
# When you promote a new model, the URI doesn't change. Production picks up
# the new version on its next reload.
champion = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
challenger = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@challenger")

champ_auc = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])
chal_auc  = roc_auc_score(y_test, challenger.predict_proba(X_test)[:, 1])
print(f"\nchampion ROC-AUC:   {champ_auc:.4f}")
print(f"challenger ROC-AUC: {chal_auc:.4f}")


# ───────────────────────────────────────────────────────────
# A real-world workflow
# ───────────────────────────────────────────────────────────
# 1. Researchers train candidate models, log them with run_name + tags.
# 2. CI evaluates each new candidate on a held-out test set.
# 3. If the candidate beats champion by enough on production-relevant
#    metrics, CI registers it and sets @challenger.
# 4. Shadow traffic compares champion vs challenger live.
# 5. If challenger holds up after N days, an operator (or auto-policy)
#    promotes it: client.set_registered_model_alias(name, "champion", v_new).
# 6. Old champion stays in the registry as a previous version — instant
#    rollback if anything regresses.
