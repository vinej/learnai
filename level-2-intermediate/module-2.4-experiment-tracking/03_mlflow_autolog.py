"""
03 — MLflow autolog: zero-effort tracking

For most popular libraries, MLflow can capture everything automatically:
  • All hyperparameters of the model.
  • Training metrics at each step (when supported).
  • Final test metrics if you call estimator.score on the test set.
  • The model itself, with a signature & input example.

Available `*.autolog()` integrations include:
  mlflow.sklearn.autolog()
  mlflow.xgboost.autolog()
  mlflow.lightgbm.autolog()
  mlflow.tensorflow.autolog()
  mlflow.pytorch.autolog()
  mlflow.transformers.autolog()
  mlflow.openai.autolog()

Run: python 03_mlflow_autolog.py
"""

import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split


mlflow.set_experiment("03_autolog")
mlflow.sklearn.autolog()                 # one line; every fit() now gets tracked


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0,
)


# ───────────────────────────────────────────────────────────
# A single fit — fully tracked, no logging code
# ───────────────────────────────────────────────────────────
with mlflow.start_run(run_name="logistic-default"):
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X_train, y_train)
    # autolog already captured params, training_score, model, signature.
    # You can still add your own:
    test_acc = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_acc)


# ───────────────────────────────────────────────────────────
# A GridSearchCV run — autolog captures the WHOLE search
# ───────────────────────────────────────────────────────────
# • One parent run with the search results.
# • One child run per fitted candidate (depending on autolog version).
# • Best params and CV score logged as metrics.
with mlflow.start_run(run_name="gbm-grid-search"):
    grid = {
        "n_estimators": [100, 200],
        "max_depth":    [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=0),
        grid, cv=3, n_jobs=-1, scoring="roc_auc",
    )
    gs.fit(X_train, y_train)
    print(f"best params: {gs.best_params_}")
    print(f"best CV ROC-AUC: {gs.best_score_:.4f}")


# ───────────────────────────────────────────────────────────
# Disable autolog when you want manual control again
# ───────────────────────────────────────────────────────────
mlflow.sklearn.autolog(disable=True)


# ───────────────────────────────────────────────────────────
# When manual logging beats autolog
# ───────────────────────────────────────────────────────────
# • You want to log custom metrics (business-specific F-beta, dollars saved).
# • You want artifacts that aren't models (your own diagnostic plots).
# • You're using a framework without an autolog integration.
# • You want fine-grained control over what ends up in the tracker (cost,
#   privacy, signal/noise).
#
# Many real projects use BOTH: autolog for the standard stuff, plus
# explicit log_metric/log_artifact for custom signals.


# ───────────────────────────────────────────────────────────
# Quick: how big should an experiment be?
# ───────────────────────────────────────────────────────────
# An "experiment" in MLflow groups RELATED runs that you'd compare directly.
# A typical structure:
#   - One experiment per ML problem ("loan-default-classifier")
#   - One run per hyperparameter combination, model variant, dataset version
#   - Tags to separate phases ("env=local", "env=prod-shadow")
