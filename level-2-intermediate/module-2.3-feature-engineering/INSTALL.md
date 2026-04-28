# Setup — Module 2.3 (Feature Engineering)

In classical ML, *what* you feed the model often matters more than *which* model you choose. This module is the practical catalog of transforms you'll reach for.

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

| Package              | Purpose                                              |
|----------------------|------------------------------------------------------|
| `scikit-learn`       | Most transforms (scalers, encoders, imputers, RFE)   |
| `imbalanced-learn`   | SMOTE / ADASYN / random over- and under-sampling     |
| `numpy`, `pandas`    | Data plumbing                                        |
| `matplotlib`, `seaborn` | Plotting                                          |

## 4. Run the lessons

```bash
python 01_numeric_scaling.py
python 02_skewed_distributions.py
python 03_binning_polynomials.py
python 04_categorical_encoding.py
python 05_target_encoding.py
python 06_imputation.py
python 07_imbalanced_classes.py
python 08_time_series_features.py
python 09_feature_selection.py
```

Plotting scripts save PNGs to `figures/`.

## 5. Run the exercises

```bash
python exercises/01_engineer_features.py
python exercises/02_ohe_vs_target.py
python exercises/03_imbalanced_strategies.py
python exercises/04_time_series_features.py
```

## A note on TargetEncoder

`sklearn.preprocessing.TargetEncoder` exists since scikit-learn 1.3. Older code/tutorials use `category_encoders` (an external library) — same idea, different import. We'll use the sklearn version throughout.

## Tip

When in doubt, build the dumbest-possible feature first (raw value, count, mean) and only get fancy if it doesn't move the needle. The shortest path to a strong baseline is usually 3-5 simple features done correctly.
