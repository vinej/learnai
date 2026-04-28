"""
10 — Matplotlib basics

The mental model:
  • Figure  — the whole picture (one or more axes)
  • Axes    — a single plot inside the figure (the thing you draw on)

Two API styles:
  pyplot:        plt.plot(x, y)               # implicit "current" axes
  object-oriented: fig, ax = plt.subplots()   # explicit — preferred for real code

Run: python 10_matplotlib_basics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


orders = pd.read_csv(Path(__file__).parent / "data" / "orders.csv",
                     parse_dates=["date"])
orders["revenue"] = orders["quantity"] * orders["price"]


# ───────────────────────────────────────────────────────────
# A single plot
# ───────────────────────────────────────────────────────────
x = np.linspace(0, 2 * np.pi, 100)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)", linestyle="--")
ax.set_title("sin and cos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "10a_line.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# Common plot types — one figure, four panels
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Bar
revenue_by_cat = orders.groupby("category")["revenue"].sum().sort_values()
axes[0, 0].bar(revenue_by_cat.index, revenue_by_cat.values, color="steelblue")
axes[0, 0].set_title("Revenue by category")
axes[0, 0].set_ylabel("revenue")

# Histogram
axes[0, 1].hist(orders["price"], bins=15, color="seagreen", edgecolor="white")
axes[0, 1].set_title("Distribution of unit price")
axes[0, 1].set_xlabel("price")

# Scatter
axes[1, 0].scatter(orders["quantity"], orders["revenue"],
                   c=orders["price"], cmap="viridis", alpha=0.7)
axes[1, 0].set_title("quantity vs revenue (color = price)")
axes[1, 0].set_xlabel("quantity")
axes[1, 0].set_ylabel("revenue")

# Line over time
monthly = orders.set_index("date")["revenue"].resample("ME").sum()
axes[1, 1].plot(monthly.index, monthly.values, marker="o")
axes[1, 1].set_title("Monthly revenue")
axes[1, 1].set_ylabel("revenue")
axes[1, 1].tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(out / "10b_panel.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# DataFrame.plot — a shortcut for quick exploration
# ───────────────────────────────────────────────────────────
# Pandas wraps matplotlib for one-liners. Less control, more speed.
ax = orders.groupby("product")["revenue"].sum().sort_values().plot(
    kind="barh", title="Revenue by product"
)
ax.set_xlabel("revenue")
ax.figure.tight_layout()
ax.figure.savefig(out / "10c_df_plot.png", dpi=120, bbox_inches="tight")
plt.close(ax.figure)


print("figures saved to:", out)


# ───────────────────────────────────────────────────────────
# Style tips
# ───────────────────────────────────────────────────────────
# • Always label axes and add a title. A plot without context is noise.
# • Use the OO API (fig, ax = ...) — easier to test and to compose.
# • Save with bbox_inches="tight" so labels don't get cropped.
# • For publication-quality, raise dpi to 200+ and use a vector format (.svg).
