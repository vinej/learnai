"""
Exercise 4 — Engineer features for a time-series forecast

Daily sales for 12 months. Predict tomorrow's sales given today's history.

  Model A — last value only (trivial baseline)
  Model B — last value + 7 engineered features

The script trains each on the first 9 months, tests on the last 3, and
asserts Model B beats Model A on RMSE.

Run: python exercises/04_time_series_features.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


rng = np.random.default_rng(0)


def make_series() -> pd.DataFrame:
    """365 daily values: trend + weekly seasonality + monthly seasonality + noise."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    t = np.arange(len(dates))
    trend     = 50 + t * 0.05
    weekly    = 10 * np.sin(2 * np.pi * t / 7)
    monthly   = 5  * np.sin(2 * np.pi * t / 30)
    weekend_lift = (dates.dayofweek >= 5).astype(float) * 8
    noise     = rng.normal(0, 3, len(dates))
    sales     = trend + weekly + monthly + weekend_lift + noise
    return pd.DataFrame({"date": dates, "sales": sales}).set_index("date")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 features useful for forecasting tomorrow's sales."""
    df = df.copy()

    # ▼▼▼ Engineered features ▼▼▼
    df["lag_1"]   = df["sales"].shift(1)
    df["lag_7"]   = df["sales"].shift(7)
    df["lag_14"]  = df["sales"].shift(14)
    prev = df["sales"].shift(1)
    df["roll_mean_7"]  = prev.rolling(7).mean()
    df["roll_std_7"]   = prev.rolling(7).std()
    df["dow"]          = df.index.dayofweek
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    # ▲▲▲

    return df


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    df = engineer(make_series()).dropna()

    cutoff = df.index[int(len(df) * 0.75)]
    train = df.loc[:cutoff]
    test  = df.loc[cutoff:]
    print(f"train {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
    print(f"test  {len(test)}  days ({test.index.min().date()} to {test.index.max().date()})")

    # Model A — last-value-only baseline
    pred_a = test["lag_1"].values        # already shifted
    rmse_a = rmse(test["sales"], pred_a)

    # Model B — gradient boosting on engineered features
    feature_cols = ["lag_1", "lag_7", "lag_14",
                    "roll_mean_7", "roll_std_7",
                    "dow", "is_weekend"]
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0,
    ).fit(train[feature_cols], train["sales"])
    pred_b = model.predict(test[feature_cols])
    rmse_b = rmse(test["sales"], pred_b)

    print(f"\nModel A (last-value-only) RMSE: {rmse_a:.3f}")
    print(f"Model B (engineered)      RMSE: {rmse_b:.3f}")
    print(f"improvement: {(1 - rmse_b / rmse_a) * 100:.1f}%")

    assert rmse_b < rmse_a, "engineered model should beat the trivial baseline"
    print("\nengineered features paid off ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add lag_30, expanding mean, day-of-month features. Do they help?
# • Switch to TimeSeriesSplit cross-validation to get more reliable RMSE.
# • Save residuals; check for autocorrelation (a model that has captured the
#   seasonal pattern should have residuals close to white noise).
# ─────────────────────────────────────────────────────────────────────────────
