# Module 5.1 — Foundations of Financial Time Series

**Level:** 5 — Extra
**Estimated time:** 1 week
**Builds on:** L1.4 (NumPy/Pandas), L2.1 (ML concepts)

## Goal

Get the data infrastructure and statistical vocabulary right *before* you forecast anything. Most failed forecasting projects fail here — wrong target, leaked features, naive split — not at the model.

## Topics

- Prices vs returns vs log-returns; why we model returns
- Stationarity, ADF/KPSS tests, differencing
- Autocorrelation (ACF) and partial autocorrelation (PACF)
- Seasonality and calendar effects
- Volatility clustering and fat tails
- Look-ahead bias, survivorship bias, point-in-time data
- Time-series train/val/test splits and walk-forward CV
- Fetching from `yfinance` (markets) and FRED (macro)

## Files

| File | What it covers |
|------|----------------|
| `01_data_sources.py` | Download equities/ETFs, FX, crypto, FRED macro series; cache locally |
| `02_prices_vs_returns.py` | Simple vs log returns, compounding, why we model returns |
| `03_stationarity.py` | ADF/KPSS tests, differencing, integration order |
| `04_acf_pacf_seasonality.py` | ACF/PACF plots, weekly/monthly seasonality, calendar features |
| `05_stylized_facts.py` | Volatility clustering, fat tails, leverage effect (visual + numeric) |
| `06_splits_and_leakage.py` | Time-aware splits, walk-forward CV, common look-ahead traps |
| `07_align_and_resample.py` | Aligning mixed-frequency data (daily prices + monthly CPI) |

## Exercises

1. Download SPY, BTC, EUR/USD and CPI from 2010-present, cache as Parquet.
2. Test stationarity of price, return, and log-return series for SPY. Explain.
3. Compute and plot ACF/PACF for daily returns vs daily squared returns. Explain the difference.
4. Build a `walk_forward_splits()` generator and use it to score a baseline (random-walk) model.
5. Find and fix three look-ahead bugs in the file `exercises/04_lookahead_traps.py`.

## Resources

- *Advances in Financial Machine Learning* — Marcos López de Prado (chapters 1-4)
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*, ch. 2-3
  (free online: https://otexts.com/fpp3/)
- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- `yfinance` README: https://github.com/ranaroussi/yfinance

## Checkpoint

You can: download a clean, point-in-time dataset for any equity/FX/crypto/macro target, justify why you're modeling returns, demonstrate stationarity, and run a walk-forward split that does not leak.
