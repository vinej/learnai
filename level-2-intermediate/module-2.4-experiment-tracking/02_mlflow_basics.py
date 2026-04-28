"""
02 — MLflow: the manual API in detail

Core building blocks:

  set_experiment()     — group of related runs
  start_run()          — context manager creating a single run
  log_param / params   — hyperparameters and config (strings)
  log_metric / metrics — numbers (can include `step` for curves)
  log_artifact(path)   — log any file (config, plot, model)
  log_artifacts(dir)   — log a whole directory
  set_tag / tags       — short labels for filtering ("baseline", "experiment-A")

You can also nest runs (e.g. parent = sweep, children = trials).

Run: python 02_mlflow_basics.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


mlflow.set_experiment("02_basics")


# ───────────────────────────────────────────────────────────
# A single richly-logged run
# ───────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)

with mlflow.start_run(run_name="rf-200-trees") as run:
    # Params (strings, model config)
    params = {
        "n_estimators":   200,
        "max_depth":      10,
        "random_state":   0,
        "dataset":        "breast_cancer",
        "feature_count":  X_train.shape[1],
    }
    mlflow.log_params(params)

    # Tags (short labels, useful for filtering in the UI)
    mlflow.set_tag("model_family", "random_forest")
    mlflow.set_tag("env", "laptop")

    # Train
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    ).fit(X_train, y_train)

    # Metrics
    y_pred  = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("roc_auc",  roc_auc_score(y_test, y_score))
    mlflow.log_metric("log_loss", log_loss(y_test, y_score))

    # Per-step metrics — useful for training curves (n_estimators 1..200)
    for n in [10, 50, 100, 150, 200]:
        partial = RandomForestClassifier(
            n_estimators=n, random_state=0,
        ).fit(X_train, y_train)
        acc = accuracy_score(y_test, partial.predict(X_test))
        mlflow.log_metric("acc_vs_estimators", acc, step=n)

    # Artifacts — a feature-importance plot and a JSON sidecar
    out = Path("scratch")
    out.mkdir(exist_ok=True)
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(importances)), importances)
    ax.set_title("Feature importances")
    fig.savefig(out / "importances.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out / "feature_meta.json").write_text(json.dumps({
        "feature_names": load_breast_cancer().feature_names.tolist(),
    }, indent=2))

    mlflow.log_artifact(out / "importances.png")
    mlflow.log_artifact(out / "feature_meta.json")

    # Log the model itself (so you can load it back later)
    mlflow.sklearn.log_model(model, name="model")

    print(f"logged run id: {run.info.run_id}")


# ───────────────────────────────────────────────────────────
# Nested runs — useful for sweeps
# ───────────────────────────────────────────────────────────
# Pattern: a "parent" run wraps a sweep; each trial is a "child" run.
# In the UI you can collapse / expand the sweep cleanly.
with mlflow.start_run(run_name="sweep-max-depth"):
    mlflow.log_param("sweep_kind", "depth")
    for depth in [3, 5, 8, None]:
        with mlflow.start_run(nested=True, run_name=f"depth={depth}"):
            mlflow.log_param("max_depth", depth)
            m = RandomForestClassifier(
                n_estimators=100, max_depth=depth, random_state=0,
            ).fit(X_train, y_train)
            mlflow.log_metric("accuracy",
                              accuracy_score(y_test, m.predict(X_test)))


print("\nrun `mlflow ui` and explore the runs.")


# ───────────────────────────────────────────────────────────
# What to log vs what to skip
# ───────────────────────────────────────────────────────────
# LOG every param that affects the result (random_state, splits, lr, depth).
# LOG every metric you might ever compare runs by.
# LOG enough artifacts to recreate the model output: plots, eval samples,
#     final config files, the model itself.
# DON'T log raw datasets — log a hash or a path. Logging GBs to the tracker
#     is slow and pollutes the store.
