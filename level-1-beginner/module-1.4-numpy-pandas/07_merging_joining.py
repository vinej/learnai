"""
07 — Combining DataFrames: merge, concat, join

  • merge   — SQL-style joins on column(s). Most common.
  • concat  — stack DataFrames (rows or columns). Schemas must align-ish.
  • join    — convenience for merging by INDEX.

Run: python 07_merging_joining.py
"""

from pathlib import Path

import pandas as pd


DATA = Path(__file__).parent / "data"
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["date"])
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])


# ───────────────────────────────────────────────────────────
# merge — the workhorse
# ───────────────────────────────────────────────────────────
# Each order has a customer_id. Pull in the customer's name and city.
joined = orders.merge(customers, on="customer_id", how="left")
print("orders enriched with customer info:")
print(joined.head())
print("\ncolumns:", joined.columns.tolist())


# ───────────────────────────────────────────────────────────
# how={inner | left | right | outer}
# ───────────────────────────────────────────────────────────
# Add a customer that has no orders, to see the difference.
extra = pd.concat([
    customers,
    pd.DataFrame([{"customer_id": 999, "name": "Zoe", "city": "Rome", "age": 30}]),
], ignore_index=True)

inner = orders.merge(extra, on="customer_id", how="inner")
left  = orders.merge(extra, on="customer_id", how="left")
right = orders.merge(extra, on="customer_id", how="right")
outer = orders.merge(extra, on="customer_id", how="outer", indicator=True)

print(f"\ninner rows: {len(inner)}    (only matched)")
print(f"left  rows: {len(left)}    (all orders)")
print(f"right rows: {len(right)}    (all customers — Zoe gets a row with NaNs)")
print(f"outer rows: {len(outer)}    (everyone)")
print("\nwhich rows came from where (outer):")
print(outer["_merge"].value_counts())


# ───────────────────────────────────────────────────────────
# Joining on different column names
# ───────────────────────────────────────────────────────────
renamed = customers.rename(columns={"customer_id": "cust_id"})
fixed = orders.merge(renamed, left_on="customer_id", right_on="cust_id", how="left")
print("\nleft_on / right_on:")
print(fixed[["order_id", "customer_id", "name", "city"]].head(3))


# ───────────────────────────────────────────────────────────
# concat — stack DataFrames
# ───────────────────────────────────────────────────────────
# Vertical stack (rows) — useful for combining batches/months.
january   = orders[orders["date"].dt.month == 1]
february  = orders[orders["date"].dt.month == 2]
stacked = pd.concat([january, february], ignore_index=True)
print(f"\nconcat'd rows: {len(stacked)}")

# Horizontal concat (columns) — by index alignment.
# Useful for adding precomputed feature columns.
extras = pd.DataFrame({"flag": [True, False]}, index=[0, 1])
print("\nhoriz concat (small example):")
print(pd.concat([orders.head(2)[["order_id", "product"]].reset_index(drop=True),
                 extras], axis=1))


# ───────────────────────────────────────────────────────────
# join — index-aligned shorthand
# ───────────────────────────────────────────────────────────
# Equivalent to df.merge(other, left_index=True, right_index=True)
left_df  = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
right_df = pd.DataFrame({"b": [10, 20]}, index=["x", "z"])
print("\njoin by index (left):")
print(left_df.join(right_df))


# ───────────────────────────────────────────────────────────
# Common pitfalls
# ───────────────────────────────────────────────────────────
# • Joining produces DUPLICATE ROWS when keys aren't unique on both sides.
#   Always check len(result) before/after, or use .merge(validate="1:m").
#
# • The KEY COLUMNS must have the same dtype. Joining string '101' with
#   int 101 gives an empty result silently. Cast first if needed.
#
# • For very large joins, sort first and use indexes — it's much faster.
