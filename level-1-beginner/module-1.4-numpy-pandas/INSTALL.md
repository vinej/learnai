# Setup — Module 1.4 (NumPy & Pandas)

This module is the data-handling workhorse: NumPy for arrays, Pandas for tabular data, plus Matplotlib + Seaborn for plots.

## 1. Python ≥ 3.11

See [../module-1.1-python-essentials/INSTALL.md](../module-1.1-python-essentials/INSTALL.md). On Windows use `py` if `python` isn't on PATH.

## 2. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package    | Purpose                                              |
|------------|------------------------------------------------------|
| `numpy`    | n-dim arrays, linear algebra                         |
| `pandas`   | tabular data (Series, DataFrame), CSV/JSON/Parquet  |
| `matplotlib` | plotting                                           |
| `seaborn`  | statistical plots on top of matplotlib              |
| `pyarrow`  | fast Parquet reader/writer used by Pandas           |

## 4. Sample data

Three small CSV files live in [data/](data/):

- `orders.csv`     — 30 clean orders
- `customers.csv`  — 10 customers (joined to orders by `customer_id`)
- `orders_messy.csv` — same shape as orders, but riddled with quality issues for the cleaning exercise

You don't have to do anything — the scripts read these directly.

## 5. Run the lessons

```bash
python 01_numpy_advanced.py
python 02_broadcasting.py
python 03_pandas_series_dataframe.py
python 04_reading_data.py
python 05_selection_filtering.py
python 06_aggregations_groupby.py
python 07_merging_joining.py
python 08_missing_data.py
python 09_time_series.py
python 10_matplotlib_basics.py
python 11_seaborn_basics.py
```

Plotting scripts save PNGs into `figures/`.

## 6. Run the exercises

```bash
python exercises/01_clean_messy_csv.py
python exercises/02_reproduce_aggregation.py
python exercises/03_vectorize_loop.py
python exercises/04_eda_dashboard.py
```

## Tips

- `pandas` is *huge*. Don't try to memorize the API — get fluent at the dozen patterns shown here, and look up the rest as you need them.
- The `info()` and `describe()` methods on a DataFrame are your first move on any new dataset.
- If you see `SettingWithCopyWarning`, you're modifying a *view* of a DataFrame. Use `.copy()` or `.loc[...]` to be explicit.
