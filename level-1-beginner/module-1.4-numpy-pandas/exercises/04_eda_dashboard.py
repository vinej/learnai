"""
Exercise 4 (Capstone for Module 1.4) — Build an EDA dashboard

Given orders + customers, produce a single PNG containing 5 plots that tell
a coherent story about the data:

  1. Monthly revenue (line)
  2. Revenue by category (bar)
  3. Revenue by city (bar)
  4. Distribution of order revenue (histogram)
  5. Correlation heatmap of numeric features

Each panel must have a title, axis labels, and reasonable limits.

Run: python exercises/04_eda_dashboard.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


DATA = Path(__file__).parent.parent / "data"
OUT  = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["date"])
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    df = orders.merge(customers, on="customer_id", how="left")
    df["revenue"] = df["quantity"] * df["price"]
    return df


def build_dashboard(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Monthly revenue
    monthly = df.set_index("date")["revenue"].resample("ME").sum()
    axes[0, 0].plot(monthly.index, monthly.values, marker="o", linewidth=2)
    axes[0, 0].set_title("Monthly revenue")
    axes[0, 0].set_ylabel("revenue")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # 2. Revenue by category
    cat = df.groupby("category")["revenue"].sum().sort_values()
    axes[0, 1].barh(cat.index, cat.values, color="steelblue")
    axes[0, 1].set_title("Revenue by category")
    axes[0, 1].set_xlabel("revenue")

    # 3. Revenue by city
    city = df.groupby("city")["revenue"].sum().sort_values(ascending=False)
    axes[0, 2].bar(city.index, city.values, color="seagreen")
    axes[0, 2].set_title("Revenue by city")
    axes[0, 2].set_ylabel("revenue")
    axes[0, 2].tick_params(axis="x", rotation=20)

    # 4. Distribution of order revenue
    sns.histplot(df, x="revenue", bins=15, hue="category",
                 multiple="stack", ax=axes[1, 0])
    axes[1, 0].set_title("Order revenue distribution")

    # 5. Correlation heatmap
    numeric = df[["quantity", "price", "revenue", "age"]].corr()
    sns.heatmap(numeric, annot=True, cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=axes[1, 1])
    axes[1, 1].set_title("Correlations")

    # 6. Customers by signup month — bonus, fills the 6th panel
    signups = (
        df.drop_duplicates("customer_id")
          .set_index("signup_date")
          .resample("ME").size()
    )
    axes[1, 2].plot(signups.index, signups.values, marker="o", color="purple")
    axes[1, 2].set_title("Customer signups per month")
    axes[1, 2].tick_params(axis="x", rotation=20)

    fig.suptitle("Orders dashboard", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    print(f"loaded {len(df)} rows, {df['customer_id'].nunique()} customers, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    out = OUT / "eda_dashboard.png"
    build_dashboard(df, out)
    print(f"dashboard saved to: {out}")

    # Brief written-summary stub. Actually fill this in!
    print(
        "\nTakeaways (write your own):"
        "\n  • Which category drives revenue?"
        "\n  • Is there a city imbalance?"
        "\n  • Any monthly trend?"
        "\n  • Strongest correlation? Surprising correlations?"
    )


if __name__ == "__main__":
    main()


# ── Stretch goals ────────────────────────────────────────────────────────────
# • Write a 100-word interpretation of the dashboard. Save it to `findings.md`.
# • Replace the synthetic data with a Kaggle dataset of your choosing.
# • Make the dashboard interactive with `streamlit` or `panel`.
# ─────────────────────────────────────────────────────────────────────────────
