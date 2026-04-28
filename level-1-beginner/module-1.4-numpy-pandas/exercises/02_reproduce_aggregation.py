"""
Exercise 2 — Reproduce specific aggregations

Given the cleaned `orders.csv` joined to `customers.csv`, produce the
following results. Each function is checked at the bottom.

Run: python exercises/02_reproduce_aggregation.py
"""

from pathlib import Path

import pandas as pd


DATA = Path(__file__).parent.parent / "data"


def load_joined() -> pd.DataFrame:
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["date"])
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    df = orders.merge(customers, on="customer_id", how="left")
    df["revenue"] = df["quantity"] * df["price"]
    return df


# ───────────────────────────────────────────────────────────
# (1) Top 3 customers by total revenue
# ───────────────────────────────────────────────────────────
def top_customers(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by customer name with their total revenue,
    sorted descending, top 3 only."""
    return (
        df.groupby("name")["revenue"]
          .sum()
          .sort_values(ascending=False)
          .head(3)
    )


# ───────────────────────────────────────────────────────────
# (2) Monthly revenue per category (pivot)
# ───────────────────────────────────────────────────────────
def monthly_category_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Index = month period, columns = category, values = total revenue."""
    return (
        df.assign(month=df["date"].dt.to_period("M"))
          .pivot_table(index="month", columns="category",
                       values="revenue", aggfunc="sum",
                       fill_value=0)
    )


# ───────────────────────────────────────────────────────────
# (3) Revenue by city, with order count and avg order value
# ───────────────────────────────────────────────────────────
def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: total_revenue, n_orders, avg_order_value (rounded 2dp)."""
    out = df.groupby("city").agg(
        total_revenue=("revenue", "sum"),
        n_orders=("order_id", "count"),
        avg_order_value=("revenue", "mean"),
    )
    out["avg_order_value"] = out["avg_order_value"].round(2)
    return out.sort_values("total_revenue", ascending=False)


def main():
    df = load_joined()

    print("Top 3 customers:")
    print(top_customers(df), "\n")
    print("Monthly category revenue:")
    print(monthly_category_revenue(df), "\n")
    print("City summary:")
    print(city_summary(df), "\n")

    # ── Spot checks ──
    top = top_customers(df)
    assert len(top) == 3
    assert top.is_monotonic_decreasing

    pivot = monthly_category_revenue(df)
    assert set(pivot.columns) == {"Books", "Kitchen", "Tools"}

    cities = city_summary(df)
    assert "total_revenue" in cities.columns
    assert cities["total_revenue"].sum() == round(df["revenue"].sum(), 2)

    print("all checks pass ✅")


if __name__ == "__main__":
    main()
