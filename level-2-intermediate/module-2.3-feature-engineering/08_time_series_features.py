"""
08 — Time-series features

For temporal data, the easiest path is "regression with temporal features":
turn each timestamp into a row of features and predict the value.

Common families:

  CALENDAR FEATURES   — hour, day-of-week, month, is_holiday.
                        Captures seasonality.
  LAG FEATURES        — value t-1, t-7, t-30. The most predictive features
                        in most time-series tasks.
  ROLLING WINDOWS     — mean / std / min / max over the last N steps.
                        Smooths noise and captures recent trend.
  EXPANDING WINDOWS   — same, but cumulative from the start.
  DIFFERENCES         — value − value.shift(k). Removes trend / seasonality.

CRITICAL: every feature must be computable using ONLY data from before the
prediction time. No look-ahead. When in doubt, draw the timeline.

Run: python 08_time_series_features.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Synthetic daily sales: trend + weekly seasonality + noise
# ───────────────────────────────────────────────────────────
dates = pd.date_range("2024-01-01", periods=180, freq="D")
trend = np.linspace(50, 80, len(dates))
weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = rng.normal(0, 5, len(dates))
sales = trend + weekly + noise

df = pd.DataFrame({"date": dates, "sales": sales}).set_index("date")
print("first 5 days:")
print(df.head())


# ───────────────────────────────────────────────────────────
# Calendar features
# ───────────────────────────────────────────────────────────
df["day_of_week"]  = df.index.dayofweek           # 0 = Monday
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["month"]        = df.index.month
df["day_of_month"] = df.index.day

# Cyclical encoding — better than raw integer for periodic features.
# day_of_week: 6 (Sun) and 0 (Mon) are NEAR each other; raw encoding hides that.
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)


# ───────────────────────────────────────────────────────────
# Lag features
# ───────────────────────────────────────────────────────────
# y_(t-k) — yesterday's value, last week's value, etc.
for lag in [1, 7, 14, 28]:
    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)


# ───────────────────────────────────────────────────────────
# Rolling features
# ───────────────────────────────────────────────────────────
# IMPORTANT: shift BEFORE rolling so the window doesn't include today's value.
prev = df["sales"].shift(1)                 # exclude today
df["rolling_mean_7"]  = prev.rolling(7).mean()
df["rolling_std_7"]   = prev.rolling(7).std()
df["rolling_max_28"]  = prev.rolling(28).max()
df["expanding_mean"]  = prev.expanding(min_periods=1).mean()


# ───────────────────────────────────────────────────────────
# Difference features
# ───────────────────────────────────────────────────────────
df["diff_1"] = df["sales"].diff(1)        # day-over-day change
df["diff_7"] = df["sales"].diff(7)        # week-over-week change


# ───────────────────────────────────────────────────────────
# Inspect
# ───────────────────────────────────────────────────────────
print("\nfeatures (sample row near the end):")
print(df.tail(1).T.to_string())


# ───────────────────────────────────────────────────────────
# Visualize
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(df.index, df["sales"], label="sales")
axes[0].plot(df.index, df["rolling_mean_7"], label="7-day rolling mean")
axes[0].set_title("Raw series + smoothed")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df["sales_lag_7"], label="lag 7", alpha=0.7)
axes[1].plot(df.index, df["sales"], label="actual", alpha=0.7)
axes[1].set_title("Original vs 1-week lag")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(df.index, df["diff_1"])
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_title("First difference (day-over-day change)")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "08_time_series.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '08_time_series.png'}")


# ───────────────────────────────────────────────────────────
# Train/test split for time series
# ───────────────────────────────────────────────────────────
# DON'T shuffle. Use the FIRST chunk as training, the LATER chunk as test.
# sklearn's TimeSeriesSplit does this for cross-validation.
n = len(df)
df = df.dropna()                             # drop the rows where lags are NaN
train = df.iloc[:int(n * 0.8)]
test  = df.iloc[int(n * 0.8):]
print(f"\ntrain {len(train)} days, test {len(test)} days")


# ───────────────────────────────────────────────────────────
# Common mistakes
# ───────────────────────────────────────────────────────────
# • Including the CURRENT value in a rolling window or expanding mean
#   (look-ahead leakage). Always shift first.
# • Random train/test split — destroys temporal order.
# • Filling lag NaNs with 0 instead of dropping — biases the model.
# • Using MAPE on values that can hit zero — explodes. Use MAE or RMSE.
