"""
06 — Aggregations: groupby, agg, pivot_table

`groupby` is the engine of analytical pandas. The pattern:

    df.groupby(key)[columns].agg(...)

works the same way as SQL's `GROUP BY`.

Run: python 06_aggregations_groupby.py
"""

from pathlib import Path

import pandas as pd


orders = pd.read_csv(Path(__file__).parent / "data" / "orders.csv",
                     parse_dates=["date"])
orders["revenue"] = orders["quantity"] * orders["price"]


# ───────────────────────────────────────────────────────────
# groupby on one column
# ───────────────────────────────────────────────────────────
revenue_by_category = orders.groupby("category")["revenue"].sum()
print("revenue by category:")
print(revenue_by_category)

print("\norders per category:")
print(orders.groupby("category").size())

# .agg with a list → multiple stats per column
stats = orders.groupby("category")["revenue"].agg(["sum", "mean", "count", "max"])
print("\nstats per category:")
print(stats)


# ───────────────────────────────────────────────────────────
# groupby on multiple columns
# ───────────────────────────────────────────────────────────
monthly_category = (
    orders
    .groupby([orders["date"].dt.to_period("M"), "category"])["revenue"]
    .sum()
    .unstack(fill_value=0)
)
print("\nmonthly revenue by category (months as rows, categories as cols):")
print(monthly_category)


# ───────────────────────────────────────────────────────────
# Different aggregations per column with a dict
# ───────────────────────────────────────────────────────────
per_customer = orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_revenue=("revenue", "sum"),
    avg_order_value=("revenue", "mean"),
    distinct_products=("product", "nunique"),
)
print("\nper-customer summary:")
print(per_customer.sort_values("total_revenue", ascending=False).head())


# ───────────────────────────────────────────────────────────
# transform — keep original shape
# ───────────────────────────────────────────────────────────
# Adds a column that's the GROUP value broadcast back to every row.
orders["category_total"] = orders.groupby("category")["revenue"].transform("sum")
orders["pct_of_category"] = orders["revenue"] / orders["category_total"]
print("\nrevenue + category share:")
print(orders[["product", "revenue", "category", "pct_of_category"]].head())


# ───────────────────────────────────────────────────────────
# pivot_table — Excel-style cross-tabulation
# ───────────────────────────────────────────────────────────
pivot = orders.pivot_table(
    index="category",
    columns=orders["date"].dt.month,
    values="revenue",
    aggfunc="sum",
    fill_value=0,
    margins=True,            # adds a "total" row and column
)
print("\npivot table (rows=category, cols=month):")
print(pivot)


# ───────────────────────────────────────────────────────────
# When NOT to groupby
# ───────────────────────────────────────────────────────────
# • For row-wise transforms with no grouping → use vectorized ops directly.
# • For very-grouped, very-fast workloads on huge data → look at polars,
#   DuckDB, or Spark. Pandas can be slow above ~10M rows.
