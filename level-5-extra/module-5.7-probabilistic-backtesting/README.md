# Module 5.7 — Probabilistic Forecasting & Backtesting

**Level:** 5 — Extra
**Estimated time:** 1 week
**Builds on:** L2.1, L4.6, all prior 5.x

## Goal

Honest forecasting is probabilistic forecasting. Point forecasts are useful as defaults but the interesting questions — risk, sizing, scenario analysis — need calibrated distributions. Then: backtest the resulting strategy, both as a forecasting exercise (CRPS, MASE) and a trading exercise (Sharpe, drawdown).

## Topics

- Forecasting metrics: MAE, RMSE, MAPE, sMAPE, MASE, RMSSE, CRPS
- Pinball / quantile loss, coverage diagrams
- **Conformal prediction** (split, full, adaptive) — distribution-free PIs
- **CQR** (Conformalized Quantile Regression) — best of both worlds
- Walk-forward backtesting protocols (anchored vs rolling)
- From forecast to trade: position sizing, risk constraints
- Trading metrics: Sharpe, Sortino, max drawdown, hit rate, expectancy
- Robustness checks: regime breakdowns, transaction costs, slippage

## Files

| File | What it covers |
|------|----------------|
| `01_metrics.py` | Implementations: MAE, RMSE, MAPE, sMAPE, MASE, RMSSE |
| `02_crps_pinball.py` | CRPS for distributional forecasts; pinball loss with coverage diagrams |
| `03_split_conformal.py` | Vanilla split conformal prediction on regression |
| `04_cqr.py` | Conformalized Quantile Regression on top of LGB quantile (5.3) |
| `05_adaptive_conformal.py` | Adaptive conformal under distribution shift |
| `06_walk_forward_backtest.py` | A clean walk-forward backtest scaffold |
| `07_forecast_to_trade.py` | Position sizing from forecast + uncertainty (Kelly-like, capped) |
| `08_trading_metrics.py` | Sharpe, Sortino, drawdown, hit rate; quantstats one-liner |

## Exercises

1. Implement MASE and RMSSE; verify on a synthetic dataset against scikit-learn / sktime.
2. Compare 80% PI coverage and width across {LGB quantile, Chronos samples, conformal-on-LGB-point}.
3. Apply CQR on top of LGB quantile (5.3); show empirical coverage now hits the nominal level.
4. Build an end-to-end walk-forward backtest: SPY 5-day forecast → position → P&L → Sharpe.
5. Run a regime-stratified backtest: split holdout by VIX quartile and report metrics per quartile.

## Resources

- *Algorithmic Learning in a Random World* — Vovk, Gammerman, Shafer (conformal foundations)
- Romano, Patterson, Candès 2019 — "Conformalized Quantile Regression"
- Gibbs & Candès 2021 — "Adaptive Conformal Inference Under Distribution Shift"
- Hyndman & Athanasopoulos — chapters 5-6 (evaluation, CV)
- `mapie`: https://mapie.readthedocs.io/
- `quantstats`: https://github.com/ranaroussi/quantstats

## Checkpoint

You can: produce a calibrated probabilistic forecast (point + interval with verified coverage), run a walk-forward backtest with realistic costs, and report both forecasting and trading metrics with regime breakdowns.
