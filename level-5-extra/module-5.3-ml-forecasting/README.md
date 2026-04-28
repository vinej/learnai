# Module 5.3 — ML Forecasting with Tabular Models

**Level:** 5 — Extra
**Estimated time:** 1 week
**Builds on:** L2.2 (scikit-learn), L2.3 (feature engineering), 5.1, 5.2

## Goal

Frame forecasting as a supervised regression problem: build features from past values + exogenous data, train gradient-boosted trees, validate with walk-forward CV. This is what wins most real-world forecasting Kaggle competitions and what most production forecast services run under the hood.

## Topics

- "Direct" vs "recursive" multi-step forecasting strategies
- Lag, rolling, expanding, and difference features
- Calendar features (without leakage)
- Exogenous features: macro, sector relatives, volatility regimes
- Target transforms: differencing, log, normalization
- Walk-forward CV with `TimeSeriesSplit` and Optuna for HPO
- Quantile regression for direct interval forecasting
- Recursive vs direct forecasting; pros/cons
- The `mlforecast` library — a clean wrapper over the above

## Files

| File | What it covers |
|------|----------------|
| `01_feature_engineering.py` | Lag, rolling, difference, calendar, regime features |
| `02_xgboost_baseline.py` | XGBoost regressor on a 1-day-ahead return target |
| `03_lightgbm_quantile.py` | LightGBM with `objective="quantile"` for prediction intervals |
| `04_direct_vs_recursive.py` | Multi-step forecasting strategies compared |
| `05_walk_forward_cv.py` | Honest CV with `TimeSeriesSplit`, no leakage |
| `06_optuna_hpo.py` | Bayesian HPO with Optuna and pruning |
| `07_mlforecast_pipeline.py` | Tidy pipeline using Nixtla's `mlforecast` |

## Exercises

1. Engineer 30+ features for SPY 1-day-ahead direction prediction; pick the top 10 by SHAP.
2. Train an LGB quantile regressor (q=0.1, 0.5, 0.9); check empirical coverage on holdout.
3. Compare direct vs recursive 5-day-ahead returns. Why does direct usually win?
4. Build an Optuna study with walk-forward CV and pruning; report best params.
5. Use `mlforecast` to fit one model across 5 ETFs (panel) and compare to per-ticker models.

## Resources

- Nixtla `mlforecast`: https://nixtlaverse.nixtla.io/mlforecast/
- *Time Series Forecasting in Python* — Marco Peixeiro
- Kaggle M5 Accuracy winners' writeups (LightGBM-heavy)
- SHAP docs: https://shap.readthedocs.io/

## Checkpoint

You can: build leakage-free features, train an XGB/LGB regressor, run walk-forward HPO, and produce calibrated quantile forecasts.
