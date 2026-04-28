# Setup — Module 2.2 (Scikit-learn)

This module covers the model catalog you'll reach for daily on tabular data, plus Pipelines and hyperparameter search.

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

| Package         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `scikit-learn`  | The catalog: linear, trees, KNN, SVM, clustering, PCA, pipelines |
| `xgboost`       | Gradient boosted trees (very strong on tabular)  |
| `lightgbm`      | Faster gradient boosting alternative             |
| `optuna`        | Modern hyperparameter optimization (Bayesian/TPE) |
| `numpy`, `pandas`, `matplotlib`, `seaborn` | Data + plotting |
| `joblib`        | Serialize trained models to disk                 |

XGBoost and LightGBM ship native binaries — installs may take a minute on slower connections. If the wheel install fails (rare), upgrade pip: `pip install --upgrade pip`.

## 4. Run the lessons

```bash
python 01_linear_models.py
python 02_trees.py
python 03_xgboost_lightgbm.py
python 04_knn_svm.py
python 05_clustering.py
python 06_pca_dimreduction.py
python 07_pipelines.py
python 08_hyperparameter_search.py
python 09_model_persistence.py
```

Plotting scripts save PNGs to `figures/`. Saved models go to `models/`.

## 5. Run the exercises

```bash
python exercises/01_california_housing.py
python exercises/02_wine_pipeline.py
python exercises/03_optuna_xgboost.py
python exercises/04_kmeans_pca.py
```

## Tips

- The "right" model depends on the data shape: linear/regularized for high-dim or small-n; tree ensembles for messy tabular; KNN/SVM rarely win in practice but are great teaching tools.
- For tabular data, GBM (XGBoost/LightGBM/CatBoost) beats deep learning ~95% of the time. Don't reach for a neural net first.
- Pipelines aren't optional — they're the difference between a notebook and a model you can ship.
