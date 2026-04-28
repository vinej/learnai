"""
03 — Pandas: Series and DataFrame

Two core types:
  • Series    — a 1-D labeled array (think: a NumPy array + an index)
  • DataFrame — a 2-D labeled table (think: a dict of Series sharing one index)

Run: python 03_pandas_series_dataframe.py
"""

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────
# Series
# ───────────────────────────────────────────────────────────
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"], name="amount")
print("Series:")
print(s)
print("\ndtype:", s.dtype)
print("name:",  s.name)
print("index:", s.index.tolist())
print("values:", s.values)

# Label-based and position-based indexing
print("\ns['b']:    ", s["b"])
print("s.iloc[1]: ", s.iloc[1])
print("s[['a','c']]:")
print(s[["a", "c"]])

# Vectorized math
print("\ns * 2:")
print(s * 2)


# ───────────────────────────────────────────────────────────
# DataFrame — three ways to construct one
# ───────────────────────────────────────────────────────────
# 1) From a dict of lists/arrays (each key becomes a column)
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol", "Dave"],
    "age":   [32, 45, 28, 51],
    "city":  ["Paris", "London", "Paris", "Berlin"],
    "score": [88.5, 72.0, 95.0, 60.5],
})
print("\nDataFrame from dict:")
print(df)

# 2) From a list of dicts (each dict becomes a row)
rows = [
    {"name": "Alice", "age": 32},
    {"name": "Bob",   "age": 45},
]
print("\nDataFrame from list of dicts:")
print(pd.DataFrame(rows))

# 3) From a NumPy array + column names
arr = np.arange(12).reshape(4, 3)
print("\nDataFrame from NumPy:")
print(pd.DataFrame(arr, columns=["a", "b", "c"]))


# ───────────────────────────────────────────────────────────
# Inspection
# ───────────────────────────────────────────────────────────
print("\nshape:", df.shape)
print("columns:", df.columns.tolist())
print("dtypes:")
print(df.dtypes)

print("\nhead(2):")
print(df.head(2))
print("\ntail(2):")
print(df.tail(2))

print("\ninfo():")
df.info()

print("\ndescribe() (numeric columns only):")
print(df.describe())


# ───────────────────────────────────────────────────────────
# Adding / removing columns
# ───────────────────────────────────────────────────────────
df["bonus"]  = df["score"] * 0.1                 # vectorized: applied row-wise
df["adult"]  = df["age"] >= 18
df = df.assign(name_upper=df["name"].str.upper())   # chainable, immutable style
print("\nwith new columns:")
print(df)

df_drop = df.drop(columns=["bonus", "adult"])
print("\nafter drop:")
print(df_drop.head())


# ───────────────────────────────────────────────────────────
# Renaming
# ───────────────────────────────────────────────────────────
df_renamed = df.rename(columns={"score": "exam_score", "age": "years"})
print("\nrenamed columns:", df_renamed.columns.tolist())


# ───────────────────────────────────────────────────────────
# The index
# ───────────────────────────────────────────────────────────
# Every DataFrame has an index. Default: 0..n-1. You can replace it.
df_indexed = df.set_index("name")
print("\nindexed by 'name':")
print(df_indexed)
print("\nlook up Alice:", df_indexed.loc["Alice", "age"])

# Get the default int index back
print("\nreset_index:")
print(df_indexed.reset_index().head(3))
