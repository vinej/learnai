"""
Exercise 2 — Diagnose overfitting / underfitting from learning curves

Three models are trained on the same dataset. For each, a learning curve
is computed and saved as a PNG. Your job:

  1. Run this script.
  2. Open `figures/exercise_02_learning_curves.png`.
  3. For each panel, decide: UNDERFIT, OVERFIT, or GOOD FIT — and why.
  4. Suggest one concrete fix for each diagnosed problem.

The script also prints train/validation MSE so you can sanity-check.

Run: python exercises/02_overfitting_diagnosis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


rng = np.random.default_rng(0)


def true_function(x):
    return np.sin(2 * np.pi * x)


n = 100
X = rng.uniform(0, 1, size=(n, 1))
y = true_function(X.ravel()) + rng.normal(0, 0.2, size=n)


models = {
    "Model 1": LinearRegression(),                                     # very simple
    "Model 2": make_pipeline(PolynomialFeatures(20), LinearRegression()),  # very flexible
    "Model 3": make_pipeline(PolynomialFeatures(4), Ridge(alpha=0.5)),  # in between
}


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, model) in zip(axes, models.items()):
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="neg_mean_squared_error",
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse   = -val_scores.mean(axis=1)

    ax.plot(sizes, train_mse, "o-", label="train")
    ax.plot(sizes, val_mse,   "o-", label="validation")
    ax.set_title(name)
    ax.set_xlabel("training-set size")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"{name}:  final train MSE = {train_mse[-1]:.3f},  "
          f"final val MSE = {val_mse[-1]:.3f}")


fig.tight_layout()
out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig_path = out / "exercise_02_learning_curves.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {fig_path}")


# ───────────────────────────────────────────────────────────
# Your diagnoses (fill in)
# ───────────────────────────────────────────────────────────
#
# Model 1 — diagnosis: ______________________  fix: ______________________
# Model 2 — diagnosis: ______________________  fix: ______________________
# Model 3 — diagnosis: ______________________  fix: ______________________
#
# Reminders:
#   UNDERFIT  → both errors stay HIGH and CONVERGE close together.
#   OVERFIT   → train error LOW, validation error MUCH HIGHER. Persistent gap.
#   GOOD FIT  → both converge to a low value with a small gap.
