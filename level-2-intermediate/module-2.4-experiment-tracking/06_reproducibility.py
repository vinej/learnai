"""
06 — Reproducibility

If you can't recreate yesterday's number on today's machine, you don't
really have a result — you have an anecdote.

The four pillars of reproducibility:

  1. SEEDS         — pin every source of randomness (Python, NumPy, frameworks).
  2. CODE VERSION  — git commit hash captured per run.
  3. ENVIRONMENT   — pin library versions in a lockfile.
  4. DATA VERSION  — hash or version your input data.

This file demonstrates 1 and gives recipes for 2-4.

Run: python 06_reproducibility.py
"""

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ───────────────────────────────────────────────────────────
# 1. Seeds — set them ALL
# ───────────────────────────────────────────────────────────
def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # If you use these later:
    #   import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    #   torch.backends.cudnn.deterministic = True
    #   torch.backends.cudnn.benchmark = False
    #   import tensorflow as tf; tf.random.set_seed(seed)
    # For sklearn estimators, pass random_state= explicitly.


set_seeds(42)


# Train twice with the same seed — outputs should match exactly.
def train_once():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1).fit(
        X_train, y_train
    )
    return accuracy_score(y_test, model.predict(X_test))


a, b = train_once(), train_once()
print(f"two runs with the same seed: {a:.6f} vs {b:.6f}  → match: {a == b}")


# ───────────────────────────────────────────────────────────
# 2. Code version — capture the git commit
# ───────────────────────────────────────────────────────────
def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


print(f"git commit: {git_commit() or '(not a git repo)'}")
print("→ MLflow logs this automatically as `mlflow.source.git.commit`.")


# ───────────────────────────────────────────────────────────
# 3. Environment — what's installed?
# ───────────────────────────────────────────────────────────
print(f"\nPython: {sys.version.split()[0]} on {platform.platform()}")
print(f"sklearn: {sklearn.__version__}")
print(f"numpy:   {np.__version__}")

# To pin: capture a lockfile from your venv. Three popular options:
#   pip freeze > requirements.txt          # quick
#   pip-compile requirements.in            # via pip-tools, repeatable
#   uv pip compile pyproject.toml -o ...   # uv (very fast)
#
# For ULTIMATE reproducibility, use Docker:
#   FROM python:3.11-slim
#   COPY requirements.txt .
#   RUN pip install -r requirements.txt
# That captures the OS too.


# ───────────────────────────────────────────────────────────
# 4. Data version — at minimum, hash it
# ───────────────────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Hash this file as a stand-in. In practice, hash your CSV / Parquet inputs.
self_hash = file_sha256(Path(__file__))
print(f"\nfile hash (sha256, first 16): {self_hash[:16]}...")
print("→ log the hash with each run so you can detect data drift")


# ───────────────────────────────────────────────────────────
# Beyond seeds: deterministic operations
# ───────────────────────────────────────────────────────────
# Some operations are non-deterministic by default for performance:
#   • GPU CUDA kernels (atomic add, float reductions).
#   • Parallel tree fits in sklearn (n_jobs > 1) on some versions.
#   • Multiprocessing — process startup order varies.
#
# Trade-offs:
#   • Force determinism → reproducibility ↑, speed ↓.
#   • Accept slight noise → speed ↑, but always confirm differences are
#     within a noise floor by running multiple seeds.


# ───────────────────────────────────────────────────────────
# Reproducibility checklist
# ───────────────────────────────────────────────────────────
# [ ] PYTHONHASHSEED set (env var, before Python starts ideally)
# [ ] Every random_state= explicit in sklearn / numpy / torch
# [ ] random / np.random seeded
# [ ] Lockfile committed (requirements.txt / poetry.lock / uv.lock)
# [ ] Git commit hash recorded with each run
# [ ] Data hash logged with each run
# [ ] Hardware noted (CPU, GPU model, OS version)
# [ ] (Optional) Dockerfile or environment.yml committed
