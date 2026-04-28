# Setup — Module 2.4 (Experiment Tracking)

After ~10 experiments, "I had a 0.92 model last week" stops being recoverable. This module covers the discipline that scales.

We'll use **MLflow** as the primary tool — it works fully offline against local files. **Weights & Biases** is mentioned for comparison; it requires an account and online auth, so the runnable lessons stick to MLflow.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

## 2. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `mlflow`       | Experiment tracking + model registry             |
| `scikit-learn` | Models for the examples                          |
| `xgboost`      | A second model framework (autolog support)       |
| `optuna`       | Hyperparameter sweep used in exercise 1          |
| `numpy`, `pandas`, `matplotlib` | Data and plotting              |

## 4. View the MLflow UI

After running any tracking script, in another terminal:

```bash
# from this folder, after a script has created mlruns/
mlflow ui
# then open http://127.0.0.1:5000 in your browser
```

The UI shows runs, parameters, metrics, plots, and lets you compare runs side-by-side.

## 5. Run the lessons

```bash
python 01_why_track.py
python 02_mlflow_basics.py
python 03_mlflow_autolog.py
python 04_compare_runs.py
python 05_model_registry.py
python 06_reproducibility.py
```

## 6. Run the exercises

```bash
python exercises/01_track_a_sweep.py
python exercises/02_promote_best_model.py
python exercises/03_reproduce_run.py
python exercises/04_make_repro.py
```

## Where do MLflow files live?

By default, MLflow writes to `./mlruns/` (run metadata) and `./mlartifacts/` (logged files). Both are created automatically. Add them to `.gitignore` — they're huge and not source.

## Tip

Every metric you'd want to inspect in 6 months — **log it**. Disk is cheap. Future-you scrolling through hand-written notes is not.
