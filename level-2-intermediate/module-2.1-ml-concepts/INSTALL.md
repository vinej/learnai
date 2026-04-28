# Setup — Module 2.1 (ML Concepts)

This module is the conceptual foundation for everything that follows. The code uses **scikit-learn** to illustrate concepts; the math and intuition are the actual lesson.

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
| `numpy`        | Arrays, linear algebra                           |
| `pandas`       | Tabular data                                     |
| `scikit-learn` | Classical ML — models, splits, metrics, pipelines |
| `matplotlib`   | Plotting                                         |
| `seaborn`      | Statistical plots                                |

You don't need to download data — scikit-learn ships several toy datasets and can synthesize larger ones with `make_classification`, `make_regression`, etc.

## 4. Run the lessons

```bash
python 01_ml_categories.py
python 02_train_val_test.py
python 03_bias_variance.py
python 04_regularization.py
python 05_classification_metrics.py
python 06_regression_metrics.py
python 07_baselines.py
python 08_data_leakage.py
```

Plotting scripts save PNGs to `figures/`.

## 5. Run the exercises

```bash
python exercises/01_metrics_workshop.py
python exercises/02_overfitting_diagnosis.py
python exercises/03_baseline_battle.py
python exercises/04_find_the_leakage.py
```

## Tip

When in doubt about a concept, change a number in the code (sample size, polynomial degree, regularization strength, class balance) and rerun. Watching how the metrics shift teaches more than any explanation.
