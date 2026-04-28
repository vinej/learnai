"""
Exercise 1 — Engineer 10+ features and measure the lift

Start with raw orders+customers data (synthetic, generated below). Build
two models:

  Model A — uses ONLY a couple of raw columns.
  Model B — uses 10+ engineered features you add.

Measure the lift in test-set R². The script asserts Model B beats Model A.

Run: python exercises/01_engineer_features.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


rng = np.random.default_rng(0)


def generate_orders(n: int = 5000) -> pd.DataFrame:
    """Generate orders data with rich underlying signal."""
    customer_id = rng.integers(1000, 1500, n)               # ~500 customers
    days_offset = rng.integers(0, 365, n)
    date = pd.to_datetime("2024-01-01") + pd.to_timedelta(days_offset, "D")

    category = rng.choice(["Tools", "Books", "Kitchen", "Apparel"], n,
                          p=[0.4, 0.25, 0.2, 0.15])
    quantity = rng.integers(1, 8, n)

    # Price varies by category
    base_price = {"Tools": 25, "Books": 12, "Kitchen": 18, "Apparel": 35}
    price = (np.array([base_price[c] for c in category])
             + rng.normal(0, 5, n)).clip(min=1)

    # Discount: more on weekends, varies by customer
    weekday = date.dayofweek.values
    discount_pct = 0.05 + 0.05 * (weekday >= 5) + rng.beta(2, 8, n) * 0.2

    # The target: revenue (with non-trivial structure on quantity * price * (1 - discount))
    revenue = quantity * price * (1 - discount_pct) + rng.normal(0, 2, n)

    return pd.DataFrame({
        "customer_id": customer_id,
        "date": date,
        "category": category,
        "quantity": quantity,
        "price": price,
        "discount_pct": discount_pct,
        "revenue": revenue,
    })


def model_baseline(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Use only quantity and price — a deliberately weak baseline."""
    features = ["quantity", "price"]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=0, n_estimators=200)),
    ])
    pipe.fit(train[features], train["revenue"])
    pred = pipe.predict(test[features])
    return r2_score(test["revenue"], pred)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add 10+ derived features. EDIT to try your own ideas."""
    df = df.copy()

    # ▼▼▼ Your engineered features ▼▼▼

    # 1-2. Calendar features
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 3. Month
    df["month"] = df["date"].dt.month

    # 4. Discounted unit price
    df["effective_price"] = df["price"] * (1 - df["discount_pct"])

    # 5. Total before discount
    df["gross"] = df["quantity"] * df["price"]

    # 6. Implied discount in absolute units
    df["abs_discount"] = df["price"] * df["discount_pct"]

    # 7-8. Customer-level aggregates (computed only on TRAIN to avoid leakage)
    customer_avg_q = df.groupby("customer_id")["quantity"].transform("mean")
    df["customer_avg_quantity"] = customer_avg_q

    customer_orders = df.groupby("customer_id")["customer_id"].transform("count")
    df["customer_n_orders"] = customer_orders

    # 9. Category-level average price
    df["category_mean_price"] = df.groupby("category")["price"].transform("mean")

    # 10. Price relative to category
    df["price_vs_category"] = df["price"] / df["category_mean_price"]

    # 11. log(quantity)
    df["log_quantity"] = np.log1p(df["quantity"])

    # ▲▲▲ End of engineered features ▲▲▲

    return df


def model_engineered(train: pd.DataFrame, test: pd.DataFrame) -> float:
    numeric = ["quantity", "price", "discount_pct", "effective_price", "gross",
               "abs_discount", "customer_avg_quantity", "customer_n_orders",
               "category_mean_price", "price_vs_category", "log_quantity",
               "dayofweek", "is_weekend", "month"]
    categorical = ["category"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale",  StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical),
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=4)),
    ])
    pipe.fit(train[numeric + categorical], train["revenue"])
    pred = pipe.predict(test[numeric + categorical])
    return r2_score(test["revenue"], pred)


def main():
    raw = generate_orders()
    train_raw, test_raw = train_test_split(raw, test_size=0.25, random_state=0)
    print(f"train {len(train_raw)}, test {len(test_raw)}")

    # Baseline (raw features only)
    r2_a = model_baseline(train_raw, test_raw)
    print(f"\nModel A (2 raw features):    R² = {r2_a:.3f}")

    # Engineered features. Engineer using BOTH train and test, but the
    # aggregate features only use training-side customers/categories
    # since we engineer per-DataFrame independently. (For a real project,
    # use a Pipeline with custom transformers to avoid leakage.)
    train_eng = engineer(train_raw)
    test_eng  = engineer(test_raw)

    r2_b = model_engineered(train_eng, test_eng)
    print(f"Model B ({11 + 1} engineered): R² = {r2_b:.3f}")
    print(f"lift: {(r2_b - r2_a) * 100:.1f} percentage points")

    assert r2_b > r2_a + 0.05, "engineered model should beat baseline by >5pp"
    print("\nfeature engineering paid off ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add target-encoding for `customer_id` (high cardinality).
# • Add interaction features: quantity × is_weekend, price × category.
# • Plot feature importances and sort your engineered features by usefulness.
# ─────────────────────────────────────────────────────────────────────────────
