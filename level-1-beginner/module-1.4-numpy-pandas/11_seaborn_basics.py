"""
11 — Seaborn: statistical plots on top of matplotlib

Seaborn ("sns") makes statistical visualizations easy and beautiful. It
operates directly on DataFrames — you pass column NAMES, not arrays.

Run: python 11_seaborn_basics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")     # nice default look


HERE = Path(__file__).parent
DATA = HERE / "data"

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["date"])
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
orders["revenue"] = orders["quantity"] * orders["price"]
joined = orders.merge(customers, on="customer_id", how="left")


out = HERE / "figures"
out.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# histplot — distributions
# ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(joined, x="revenue", hue="category", multiple="stack",
             bins=20, ax=ax)
ax.set_title("Revenue distribution by category")
fig.savefig(out / "11a_histplot.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# boxplot / violinplot — distributions per group
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.boxplot(joined, x="category", y="revenue", ax=axes[0])
axes[0].set_title("Boxplot: revenue per category")

sns.violinplot(joined, x="city", y="revenue", inner="quartile", ax=axes[1])
axes[1].set_title("Violinplot: revenue per city")
axes[1].tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(out / "11b_box_violin.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# Correlation heatmap
# ───────────────────────────────────────────────────────────
# Useful first step in any EDA — what features move together?
numeric_cols = ["quantity", "price", "revenue", "age"]
corr = joined[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation heatmap")
fig.tight_layout()
fig.savefig(out / "11c_heatmap.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# pairplot — every pair of numeric features
# ───────────────────────────────────────────────────────────
# Hugely useful for quick EDA. Returns a Figure, not an Axes.
g = sns.pairplot(joined[numeric_cols + ["category"]],
                 hue="category", diag_kind="hist", height=2)
g.figure.savefig(out / "11d_pairplot.png", dpi=120, bbox_inches="tight")
plt.close(g.figure)


# ───────────────────────────────────────────────────────────
# Bar / count plots
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.countplot(joined, x="city", hue="category", ax=axes[0])
axes[0].set_title("Order counts: city × category")
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(joined, x="category", y="revenue", estimator="sum", ax=axes[1])
axes[1].set_title("Total revenue per category")

fig.tight_layout()
fig.savefig(out / "11e_bar_count.png", dpi=120, bbox_inches="tight")
plt.close(fig)


print("seaborn figures saved to:", out)


# ───────────────────────────────────────────────────────────
# When to use what
# ───────────────────────────────────────────────────────────
# • Distribution of one variable        → histplot, kdeplot
# • Distribution per category           → boxplot, violinplot, stripplot
# • Two numerical variables             → scatterplot, regplot, jointplot
# • Multiple variables together         → pairplot, heatmap
# • Counts of categorical levels        → countplot, barplot
