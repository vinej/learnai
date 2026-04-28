"""
08 — Missing data (NaN, NaT, None)

Pandas marks missing values as `NaN` (numbers), `NaT` (datetimes), or `None`
(objects). Most operations skip them by default.

Tools:
  isna / notna       — detect missing
  dropna             — remove rows/cols with missing values
  fillna             — replace missing
  ffill / bfill      — propagate last/next valid value
  interpolate        — fill numerically
  replace            — generic value replacement (incl. sentinel values)

Run: python 08_missing_data.py
"""

import numpy as np
import pandas as pd


df = pd.DataFrame({
    "name":  ["Alice", "Bob", None, "Dave", "Eve"],
    "age":   [32, np.nan, 28, 51, np.nan],
    "city":  ["Paris", "London", "Paris", None, "Madrid"],
    "score": [88.5, 72.0, np.nan, 60.5, 95.0],
})
print("original:")
print(df)


# ───────────────────────────────────────────────────────────
# Detecting NaNs
# ───────────────────────────────────────────────────────────
print("\nisna:")
print(df.isna())

print("\nNaN count per column:")
print(df.isna().sum())

print("\nrows with any NaN:")
print(df[df.isna().any(axis=1)])


# ───────────────────────────────────────────────────────────
# Dropping
# ───────────────────────────────────────────────────────────
print("\ndropna() — drop ANY row with NaN:")
print(df.dropna())

print("\ndropna(subset=['age']):")
print(df.dropna(subset=["age"]))

print("\ndropna(axis=1) — drop COLUMNS that have any NaN:")
print(df.dropna(axis=1))


# ───────────────────────────────────────────────────────────
# Filling
# ───────────────────────────────────────────────────────────
# A single value
print("\nfill all NaNs with 'unknown' / 0:")
print(df.fillna({"name": "unknown", "city": "unknown",
                 "age": df["age"].mean(),
                 "score": df["score"].median()}))

# ffill — forward-fill (use the last seen value). Common for time series.
ts = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan])
print("\nffill:", ts.ffill().tolist())
print("bfill:", ts.bfill().tolist())


# ───────────────────────────────────────────────────────────
# Interpolation (numeric only)
# ───────────────────────────────────────────────────────────
ts2 = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
print("\ninterpolate (linear):")
print(ts2.interpolate().tolist())


# ───────────────────────────────────────────────────────────
# replace — for sentinel values
# ───────────────────────────────────────────────────────────
# Real datasets often use -1, 999, "N/A" to mean missing. Convert to NaN
# before doing any analysis.
weird = pd.DataFrame({"x": [1, 2, -1, 4, -1], "label": ["a", "b", "N/A", "d", "N/A"]})
cleaned = weird.replace({-1: np.nan, "N/A": np.nan})
print("\nreplace sentinels:")
print(cleaned)


# ───────────────────────────────────────────────────────────
# Aggregations and NaN
# ───────────────────────────────────────────────────────────
# By default, sum/mean/etc. SKIP NaN.
print("\ndf['age'].mean()      ->", df["age"].mean())     # ignores NaN
print("df['age'].count()     ->", df["age"].count())     # non-null count
print("len(df)               ->", len(df))               # total rows
print("df['age'].sum(skipna=False) ->", df["age"].sum(skipna=False))


# ───────────────────────────────────────────────────────────
# Best practice
# ───────────────────────────────────────────────────────────
# • Decide explicitly: drop rows? impute with mean/median/mode? carry forward?
#   The right call depends on the dataset and the question.
# • Document why you chose what you did. Someone (often future-you) WILL ask.
