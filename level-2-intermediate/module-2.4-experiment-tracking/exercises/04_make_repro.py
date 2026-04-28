"""
Exercise 4 — Make an existing project reproducible

`messy_train()` below is a "vibe-coded" training function: no seeds, no
tracking, no idea what library versions you ran it with.

Your job: turn it into `repro_train()` that:
  1. Sets all relevant seeds.
  2. Logs params, metrics, and the resulting model to MLflow.
  3. Logs the python and library versions as tags.
  4. Returns a deterministic score across calls with the same seed.

The script asserts:
  • two calls with the same seed match exactly.
  • the run shows up in MLflow with the expected tags.

Run: python exercises/04_make_repro.py
"""

import os
import random
import sys

import mlflow
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


mlflow.set_experiment("ex4_repro")


# ───────────────────────────────────────────────────────────
# Before — no seeds, no tracking, no versioning
# ───────────────────────────────────────────────────────────
def messy_train() -> float:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GradientBoostingClassifier(n_estimators=200).fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# ───────────────────────────────────────────────────────────
# After — seeded and tracked
# ───────────────────────────────────────────────────────────
def repro_train(seed: int = 0) -> float:
    # ▼▼▼ Your reproducibility upgrades ▼▼▼

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    params = {
        "n_estimators":   200,
        "learning_rate":  0.1,
        "max_depth":      3,
        "random_state":   seed,
    }

    with mlflow.start_run(run_name=f"repro-seed-{seed}"):
        mlflow.log_params(params)
        mlflow.log_param("split_seed", seed)

        # Environment versions captured as tags
        mlflow.set_tag("python_version", sys.version.split()[0])
        mlflow.set_tag("sklearn_version", sklearn.__version__)
        mlflow.set_tag("numpy_version",   np.__version__)

        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        model = GradientBoostingClassifier(**params).fit(X_train, y_train)
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_roc_auc", score)
        mlflow.sklearn.log_model(model, name="model")

    return score
    # ▲▲▲ end of upgrades ▲▲▲


def main():
    a = repro_train(seed=42)
    b = repro_train(seed=42)
    print(f"two seeded runs: {a:.6f} vs {b:.6f}")
    assert a == b, "seeded runs should match exactly"

    # Sanity-check tags via MLflow
    runs = mlflow.search_runs(experiment_names=["ex4_repro"]).head(1)
    assert "tags.python_version" in runs.columns, "you forgot to log version tags"
    assert runs.iloc[0]["tags.python_version"] != "", "python_version tag is empty"

    print(f"\nlatest run captured python_version={runs.iloc[0]['tags.python_version']}, "
          f"sklearn_version={runs.iloc[0]['tags.sklearn_version']}")
    print("project is now reproducible ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Hash the input dataset and log the hash as a param.
# • Capture the git commit (subprocess git rev-parse HEAD) and log as a tag.
# • Pin the env: pip-tools or uv to produce a deterministic requirements lock.
# ─────────────────────────────────────────────────────────────────────────────
