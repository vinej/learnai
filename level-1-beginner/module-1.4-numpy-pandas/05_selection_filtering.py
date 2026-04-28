"""
05 — Selecting rows and columns, filtering with boolean masks

The mental model:
  • df[col]                       column → Series
  • df[[c1, c2]]                  multiple columns → DataFrame
  • df.loc[row_label, col_label]  LABEL-based (works on column names + index)
  • df.iloc[row_pos,   col_pos]   POSITION-based (integer indices, like NumPy)
  • df[boolean_mask]              filter rows
  • df.query("expr")              SQL-like filtering with a string

Run: python 05_selection_filtering.py
"""

from pathlib import Path

import pandas as pd


orders = pd.read_csv(Path(__file__).parent / "data" / "orders.csv",
                     parse_dates=["date"])


# ───────────────────────────────────────────────────────────
# Pick columns
# ───────────────────────────────────────────────────────────
print("just product:")
print(orders["product"].head(3))      # Series

print("\nproduct + price:")
print(orders[["product", "price"]].head(3))   # DataFrame


# ───────────────────────────────────────────────────────────
# loc vs iloc — pick BOTH dimensions
# ───────────────────────────────────────────────────────────
# .loc is INCLUSIVE on its right edge (because it's label-based).
# .iloc is EXCLUSIVE on its right edge (because it's position-based, like Python slicing).
print("\nloc[0:2, 'product':'price']:")
print(orders.loc[0:2, "product":"price"])     # rows 0,1,2  cols product..price

print("\niloc[0:3, 3:6]:")
print(orders.iloc[0:3, 3:6])                  # rows 0..2  cols 3..5


# ───────────────────────────────────────────────────────────
# Boolean filtering
# ───────────────────────────────────────────────────────────
big_orders = orders[orders["quantity"] >= 4]
print("\nbig orders (quantity >= 4):")
print(big_orders)

# Combine conditions with &  |  ~  (and, or, not). PARENTHESIZE each one!
mask = (orders["category"] == "Tools") & (orders["price"] > 20)
print("\npricey Tools:")
print(orders[mask])


# ───────────────────────────────────────────────────────────
# isin and between
# ───────────────────────────────────────────────────────────
in_kitchen_or_books = orders[orders["category"].isin(["Kitchen", "Books"])]
print("\nKitchen or Books:")
print(in_kitchen_or_books.head())

january = orders[orders["date"].between("2024-01-01", "2024-01-31")]
print("\nJanuary orders:", len(january))


# ───────────────────────────────────────────────────────────
# query() — same filters, prettier syntax
# ───────────────────────────────────────────────────────────
# Rule of thumb: query() is great for one-liners; boolean masks are easier
# when the condition is dynamic.
result = orders.query("category == 'Tools' and price > 20")
print("\nquery result:")
print(result.head())

# Reference an external variable with @:
threshold = 4.99
print("\nquery using @threshold:")
print(orders.query("price <= @threshold").head())


# ───────────────────────────────────────────────────────────
# Adding a derived column based on a filter
# ───────────────────────────────────────────────────────────
# Vectorized — no for-loop needed.
orders["revenue"] = orders["quantity"] * orders["price"]
orders["bulk"] = orders["quantity"] >= 4
print("\nwith derived columns:")
print(orders.head())


# ───────────────────────────────────────────────────────────
# A note on SettingWithCopyWarning
# ───────────────────────────────────────────────────────────
# This is the most-asked Pandas question on Stack Overflow.
# It happens when you slice a DataFrame and then ASSIGN to it:
#
#   subset = orders[orders["price"] > 10]
#   subset["foo"] = 1                       # ⚠️ may warn — modifies a view
#
# Fix: be explicit with .copy() if you intend a new DataFrame:
#
#   subset = orders[orders["price"] > 10].copy()
#   subset["foo"] = 1                       # ✓ clear intent
