"""
Exercise 1 — Clean a messy CSV

`data/orders_messy.csv` has the same shape as `orders.csv` but with realistic
data-quality problems:
  • whitespace around values
  • inconsistent date formats
  • dollar signs in prices
  • numbers written as words ("four")
  • mixed-case category & product values
  • blank cells
  • a negative quantity

Your job: clean it into a typed DataFrame that matches the schema of the
clean orders.csv. The script asserts your final dtypes and row count.

Run: python exercises/01_clean_messy_csv.py

Hints:
  • pd.read_csv has `skipinitialspace=True`
  • str.strip(), str.title()  for text cleanup
  • pd.to_datetime(format='mixed')
  • pd.to_numeric(errors='coerce')  to turn bad numbers into NaN
"""

import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent.parent
SRC  = HERE / "data" / "orders_messy.csv"


def clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)

    # ▼▼▼ Your cleaning code here ▼▼▼

    # 1. Strip whitespace everywhere
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Strip $ from price, then to numeric
    df["price"] = df["price"].astype(str).str.replace("$", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 3. Coerce quantity to numeric ("four" → NaN), then drop negatives
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df.loc[df["quantity"] < 0, "quantity"] = pd.NA

    # 4. Standardize text columns
    df["product"]  = df["product"].str.title()
    df["category"] = df["category"].str.title()

    # 5. Customer id may be missing or stringy
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")

    # 6. Mixed date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    # 7. Drop rows missing any required field
    required = ["date", "customer_id", "product", "category", "quantity", "price"]
    df = df.dropna(subset=required)

    # 8. Final dtype tightening
    df["customer_id"] = df["customer_id"].astype("int32")
    df["quantity"]    = df["quantity"].astype("int32")

    # ▲▲▲ Your cleaning code here ▲▲▲
    return df.reset_index(drop=True)


def main():
    df = clean(SRC)
    print(df)
    print("\ndtypes:")
    print(df.dtypes)

    assert df["date"].dtype.kind == "M",        "date should be datetime"
    assert df["customer_id"].dtype.kind == "i", "customer_id should be int"
    assert df["quantity"].dtype.kind == "i",    "quantity should be int"
    assert df["price"].dtype.kind == "f",       "price should be float"
    assert (df["quantity"] > 0).all(),          "no non-positive quantities"
    assert df["product"].str.istitle().all(),   "products should be Title Case"

    print(f"\nkept {len(df)} clean rows out of 20.")


if __name__ == "__main__":
    main()


# ── Reflection ───────────────────────────────────────────────────────────────
# In real projects, every dataset has its own quirks. The skill is:
#   1. Inspect first (df.head(), df.info(), df.describe(include='all'))
#   2. Document the issues you find
#   3. Decide whether to drop, impute, or correct each issue
#   4. Encode your decisions as testable assertions (like above)
# That's the difference between "wrangling" and "engineering".
# ─────────────────────────────────────────────────────────────────────────────
