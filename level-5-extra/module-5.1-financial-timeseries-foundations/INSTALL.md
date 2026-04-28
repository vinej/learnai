# Setup — Module 5.1

Inherits the Level 5 environment. If you only want to do this module:

```bash
python -m venv .venv
# activate, then
pip install numpy pandas scipy matplotlib statsmodels yfinance fredapi pandas-datareader python-dotenv pyarrow
```

Set your FRED key in `level-5-extra/.env`:

```env
FRED_API_KEY=...
```

(Free, sign up at https://fred.stlouisfed.org/docs/api/api_key.html.)

`yfinance` does not need a key.
