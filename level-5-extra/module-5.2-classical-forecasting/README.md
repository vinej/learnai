# Module 5.2 — Classical Statistical Forecasting

**Level:** 5 — Extra
**Estimated time:** 1 week
**Builds on:** L1.3 (math for AI), L2.1 (ML concepts), 5.1

## Goal

Build the statistical baselines that fancier models must beat. Despite the deep-learning era, ARIMA and ETS still win on many real datasets — especially short, clean series — and they're cheap, fast, and explainable.

## Topics

- Naive baselines: random walk, seasonal naive, drift
- Exponential smoothing (ETS): simple, Holt, Holt-Winters
- ARIMA / SARIMA: identification, fitting, diagnostics, auto-selection
- Prophet: trend + seasonality + holidays
- GARCH for volatility forecasting
- VAR for multivariate macro forecasting
- Forecast combination (simple averages outperform individual models)

## Files

| File | What it covers |
|------|----------------|
| `01_baselines.py` | Naive, seasonal-naive, drift, mean — your floor |
| `02_exponential_smoothing.py` | Simple/Holt/Holt-Winters via `statsmodels.tsa.holtwinters` |
| `03_arima_sarima.py` | Identification with ACF/PACF; `auto_arima` from `pmdarima` (or `statsforecast`) |
| `04_prophet_basics.py` | `prophet.Prophet`: changepoints, holidays, regressors |
| `05_garch_volatility.py` | `arch_model`: forecasting volatility on SPY/BTC |
| `06_var_macro.py` | VAR on a small macro panel (yield, CPI YoY, fed funds) |
| `07_forecast_combination.py` | Simple average + weighted combination by inverse MSE |

## Exercises

1. Beat the seasonal-naive baseline on monthly CPI YoY.
2. Fit ARIMA(p,d,q) by hand using ACF/PACF inspection on SPY monthly returns; compare to `auto_arima`.
3. Use Prophet on BTC daily price with a custom Bitcoin-halving regressor.
4. Forecast SPY 1-day-ahead realized volatility using GARCH(1,1) and evaluate vs realized squared return.
5. Build a VAR(p) on (CPI YoY, UNRATE, DGS10) and produce 12-month-ahead forecasts with confidence bands.

## Why classical methods still matter

- They're the only family with closed-form prediction intervals that are usually well-calibrated.
- They're nearly free to fit; you can run thousands of variants for ensembling.
- M-competition results 2018-2024 consistently show ETS + ARIMA averaged with a NN matches or beats either alone.

## Resources

- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (free): https://otexts.com/fpp3/
- `statsmodels` time-series docs: https://www.statsmodels.org/stable/tsa.html
- `statsforecast` — fast, sklearn-style API for AutoARIMA/ETS/Theta: https://github.com/Nixtla/statsforecast

## Checkpoint

You can: pick the right classical model for a given series (trend? seasonality? heteroscedasticity?), fit it, validate residuals, and produce point + interval forecasts.
