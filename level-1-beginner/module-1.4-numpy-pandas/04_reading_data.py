"""
04 — Reading and writing data

Pandas reads almost any tabular format. The two you'll use 95% of the time:
  • read_csv  — the universal lingua franca
  • read_parquet — columnar, compressed, typed → your friend for big data

Run: python 04_reading_data.py
"""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
DATA = HERE / "data"


# ───────────────────────────────────────────────────────────
# CSV
# ───────────────────────────────────────────────────────────
# Auto-detects types and the header row.
orders = pd.read_csv(DATA / "orders.csv")
print("first 5 orders:")
print(orders.head())
print("\ndtypes:")
print(orders.dtypes)

# Useful options:
#   sep           — separator (default ',', use '\t' for TSV)
#   header        — int row to use as column names; None if no header
#   names         — supply your own column names
#   parse_dates   — parse columns as datetime
#   dtype         — coerce specific columns
#   na_values     — extra strings to treat as NaN ("N/A", "missing", ...)
#   nrows         — read only first N rows (handy for huge files)
#   chunksize     — return an iterator for streaming

orders_typed = pd.read_csv(
    DATA / "orders.csv",
    parse_dates=["date"],
    dtype={"customer_id": "int32", "quantity": "int16"},
)
print("\ntyped dtypes:")
print(orders_typed.dtypes)


# ───────────────────────────────────────────────────────────
# Writing CSV
# ───────────────────────────────────────────────────────────
out = HERE / "scratch"
out.mkdir(exist_ok=True)

orders.to_csv(out / "orders_copy.csv", index=False)
# `index=False` so the row numbers aren't written as a column.
print(f"\nwrote: {out / 'orders_copy.csv'}")


# ───────────────────────────────────────────────────────────
# JSON
# ───────────────────────────────────────────────────────────
orders.head(3).to_json(out / "orders.json", orient="records", indent=2)
loaded_json = pd.read_json(out / "orders.json")
print("\nfrom JSON:")
print(loaded_json)


# ───────────────────────────────────────────────────────────
# Parquet  (requires `pyarrow`)
# ───────────────────────────────────────────────────────────
# Why Parquet? Columnar, compressed, types preserved, much faster to read
# than CSV for large data. Common output of data pipelines and feature stores.
orders_typed.to_parquet(out / "orders.parquet")
loaded_parquet = pd.read_parquet(out / "orders.parquet")
print("\nfrom Parquet:")
print(loaded_parquet.head())
print("dtypes preserved:")
print(loaded_parquet.dtypes)


# ───────────────────────────────────────────────────────────
# read_sql (briefly)
# ───────────────────────────────────────────────────────────
# Pandas can read query results from any DB supported by SQLAlchemy:
#
#     from sqlalchemy import create_engine
#     engine = create_engine("postgresql://user:pass@host/db")
#     df = pd.read_sql("SELECT * FROM orders WHERE date > '2024-01-01'", engine)
#
# We won't actually connect to a DB here, but the API is identical to read_csv.


# ───────────────────────────────────────────────────────────
# Reading "messy" CSVs — preview
# ───────────────────────────────────────────────────────────
# Real CSVs are rarely this clean. See 08_missing_data.py and the cleaning
# exercise (exercises/01_clean_messy_csv.py) for the full treatment.
messy = pd.read_csv(DATA / "orders_messy.csv")
print("\nmessy preview (note dtypes — not all numeric!):")
print(messy.dtypes)
print(messy.head())
