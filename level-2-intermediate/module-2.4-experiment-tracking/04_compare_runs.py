"""
04 — Querying and comparing runs

The UI is great for browsing. From code, you'll often want to:

  • Find the best run by some metric.
  • Pull a DataFrame of all runs for analysis.
  • Filter runs by tags or parameter values.
  • Diff two runs.

`mlflow.search_runs()` gives you everything as a Pandas DataFrame.

Run: python 04_compare_runs.py
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


EXPERIMENT = "04_compare"
mlflow.set_experiment(EXPERIMENT)


# ───────────────────────────────────────────────────────────
# Generate a few runs to compare
# ───────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)


def log_run(name, model, family):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", family)
        for k, v in model.get_params().items():
            mlflow.log_param(k, v)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("roc_auc",  roc_auc_score(y_test, y_score))


# Three model families
log_run("logreg-C-0.1", LogisticRegression(C=0.1, max_iter=2000), "linear")
log_run("logreg-C-1",   LogisticRegression(C=1.0, max_iter=2000), "linear")
log_run("logreg-C-10",  LogisticRegression(C=10,  max_iter=2000), "linear")

log_run("rf-100",  RandomForestClassifier(n_estimators=100, random_state=0), "rf")
log_run("rf-300",  RandomForestClassifier(n_estimators=300, random_state=0), "rf")

log_run("gbm",     GradientBoostingClassifier(random_state=0), "gbm")


# ───────────────────────────────────────────────────────────
# Pull all runs as a DataFrame
# ───────────────────────────────────────────────────────────
runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
print(f"\n{len(runs)} runs in experiment '{EXPERIMENT}'")
print("\ncolumns include:")
print([c for c in runs.columns if not c.startswith("params.")][:15], "...")


# ───────────────────────────────────────────────────────────
# Sort and show best run
# ───────────────────────────────────────────────────────────
top = runs.sort_values("metrics.roc_auc", ascending=False).head(3)[
    ["tags.mlflow.runName", "tags.model_family",
     "metrics.accuracy", "metrics.roc_auc"]
].rename(columns=lambda c: c.replace("tags.", "").replace("metrics.", ""))
print("\ntop 3 by ROC-AUC:")
print(top.to_string(index=False))


# ───────────────────────────────────────────────────────────
# Filter with the same syntax MLflow's UI uses
# ───────────────────────────────────────────────────────────
# Syntax: "tags.X = 'value' AND metrics.Y > 0.95"
linear_runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    filter_string="tags.model_family = 'linear' and metrics.roc_auc > 0.99",
)
print(f"\nlinear runs with ROC-AUC > 0.99: {len(linear_runs)}")


# ───────────────────────────────────────────────────────────
# Diff two runs by parameters
# ───────────────────────────────────────────────────────────
def diff_runs(run_id_a: str, run_id_b: str) -> pd.DataFrame:
    a = mlflow.get_run(run_id_a).data.params
    b = mlflow.get_run(run_id_b).data.params
    keys = sorted(set(a) | set(b))
    return pd.DataFrame([{
        "param": k, "a": a.get(k, "—"), "b": b.get(k, "—"),
    } for k in keys if a.get(k) != b.get(k)])


# Pick the two top runs and diff them
if len(runs) >= 2:
    a_id = runs.iloc[0]["run_id"]
    b_id = runs.iloc[1]["run_id"]
    print(f"\ndiff between top two runs:")
    print(diff_runs(a_id, b_id).to_string(index=False))


# ───────────────────────────────────────────────────────────
# Loading a model back from a run
# ───────────────────────────────────────────────────────────
# Each run with a logged model is reloadable:
#
#   model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
#   preds = model.predict(X_new)
#
# In file 05 we cover the model REGISTRY, which gives you stable names like
# "loan-classifier@champion" instead of opaque run IDs.
