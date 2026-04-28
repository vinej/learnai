"""
09 — Time series basics

Pandas has first-class support for dates. The key types:
  • Timestamp / DatetimeIndex      — points in time
  • Period       / PeriodIndex     — spans (e.g. 'Jan 2024')
  • Timedelta                      — durations

Most operations want a DatetimeIndex.

Run: python 09_time_series.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────
# Parsing dates
# ───────────────────────────────────────────────────────────
orders = pd.read_csv(Path(__file__).parent / "data" / "orders.csv",
                     parse_dates=["date"])
print("date dtype:", orders["date"].dtype)

# pd.to_datetime can parse many formats and is your friend for messy data
mixed = pd.Series(["2024-01-05", "2024/02/01", "Mar 15 2024"])
print("\nparsed mixed:", pd.to_datetime(mixed, format="mixed").tolist())


# ───────────────────────────────────────────────────────────
# Date components — .dt accessor
# ───────────────────────────────────────────────────────────
orders["year"] = orders["date"].dt.year
orders["month"] = orders["date"].dt.month
orders["weekday"] = orders["date"].dt.day_name()
print("\norders with date features:")
print(orders[["date", "year", "month", "weekday"]].head())


# ───────────────────────────────────────────────────────────
# Set the date as the index — unlocks resample / rolling
# ───────────────────────────────────────────────────────────
orders["revenue"] = orders["quantity"] * orders["price"]
ts = orders.set_index("date").sort_index()


# ───────────────────────────────────────────────────────────
# Resample — like groupby for time
# ───────────────────────────────────────────────────────────
# DOWN-sample: many points → fewer points (aggregate)
weekly = ts["revenue"].resample("W").sum()
print("\nweekly revenue:")
print(weekly.head())

monthly = ts["revenue"].resample("ME").sum()    # 'ME' = month end
print("\nmonthly revenue:")
print(monthly)

# UP-sample: fewer points → more points (need to fill)
daily = monthly.resample("D").ffill()
print("\nupsampled (forward-filled) head:")
print(daily.head())


# ───────────────────────────────────────────────────────────
# Rolling windows
# ───────────────────────────────────────────────────────────
# Useful for moving averages, smoothing, lag-style features.
rng = np.random.default_rng(0)
clean_daily = (
    pd.Series(rng.normal(100, 10, 90),
              index=pd.date_range("2024-01-01", periods=90, freq="D"),
              name="metric")
)
moving_7d = clean_daily.rolling(window=7).mean()
print("\n7-day rolling average (head):")
print(moving_7d.head(10))


# ───────────────────────────────────────────────────────────
# Lag / diff / shift
# ───────────────────────────────────────────────────────────
# These are essential for time-series feature engineering.
small = clean_daily.head(5)
print("\nshifts:")
print(pd.DataFrame({
    "value":      small,
    "yesterday":  small.shift(1),    # previous day's value
    "delta":      small.diff(),      # change vs. yesterday
    "pct_change": small.pct_change(),
}))


# ───────────────────────────────────────────────────────────
# Date math
# ───────────────────────────────────────────────────────────
today = pd.Timestamp("2024-04-01")
last_week = today - pd.Timedelta(days=7)
print(f"\n{today} - 7 days = {last_week}")

# Filter on a date range using .loc with strings
print("\nMarch orders:")
print(ts.loc["2024-03"][["product", "revenue"]].head())


# ───────────────────────────────────────────────────────────
# A few good defaults
# ───────────────────────────────────────────────────────────
# • Always sort the index before resample/rolling — otherwise results
#   are silently wrong.
# • Use UTC timezones for any data that crosses time zones.
# • Watch out for daylight savings; pd handles it but the math gets tricky.
