"""
Exercise 3 — Beat a baseline

You're handed the California housing dataset. Build:

  1. A trivial baseline (DummyRegressor with strategy='mean').
  2. ONE simple model of your choice (LinearRegression, Ridge, KNN,
     DecisionTreeRegressor — pick one).
  3. A more capable model (GradientBoostingRegressor or RandomForestRegressor).

Report RMSE on the test set for each, and the % improvement of each over
the baseline.

The script provides scaffolding. Fill in the TODOs.

Run: python exercises/03_baseline_battle.py
"""

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def main() -> None:
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ── 1. Baseline ──
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_rmse = rmse(y_test, baseline.predict(X_test))

    # ── 2. Simple model ──  TODO: pick & train one
    simple = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    simple.fit(X_train, y_train)
    simple_rmse = rmse(y_test, simple.predict(X_test))

    # ── 3. Stronger model ──  TODO: tune at least one hyperparameter
    strong = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=0,
    ).fit(X_train, y_train)
    strong_rmse = rmse(y_test, strong.predict(X_test))

    # ── Report ──
    print(f"{'model':<25} {'RMSE':>8} {'vs baseline':>14}")
    for name, val in [
        ("baseline (mean)", baseline_rmse),
        ("simple (Ridge)", simple_rmse),
        ("strong (GBM)",   strong_rmse),
    ]:
        improvement = (1 - val / baseline_rmse) * 100
        print(f"{name:<25} {val:>8.3f} {improvement:>13.1f}%")

    assert simple_rmse < baseline_rmse, "your simple model didn't beat the baseline!"
    assert strong_rmse < simple_rmse,   "your strong model didn't beat the simple one!"
    print("\nbaselines beaten ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Try a different "simple" model (KNeighborsRegressor, DecisionTree).
#   Which one beats Ridge most cleanly here?
# • Add cross-validation: average RMSE over 5 folds.
# • Compute MAE alongside RMSE — does the ranking change?
# ─────────────────────────────────────────────────────────────────────────────
