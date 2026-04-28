# LEVEL 5 — Extra: Capstone — Financial Forecasting (2026)

**Level:** 5 — Extra (Capstone)
**Estimated time:** 8-10 weeks
**Prerequisites:** Levels 1-4 completed (or equivalent experience)

## Goal

Apply everything from Levels 1-4 to build a multi-method financial forecaster: classical statistics, gradient-boosted trees, deep learning, time-series foundation models, and LLM-augmented analysis — combined under proper backtesting, uncertainty quantification, and a deployable serving layer.

## Why this exists

Financial forecasting is one of the few domains where every technique you've learned is genuinely useful, where the data is plentiful and free, where the ground truth arrives quickly, and where mistakes are visible. It is also a domain rich in honest difficulty: signals are noisy, regimes shift, and look-ahead bias is easy to introduce. Working through it forces you to apply L1-L4 with real discipline.

## Domain coverage

Examples and exercises focus on:
- **Equities & ETFs** — SPY, QQQ, sector ETFs (XLK, XLF, XLE), individual names (AAPL, NVDA, etc.)
- **FX** — EUR/USD, USD/JPY, DXY
- **Crypto** — BTC, ETH (24/7 markets, regime changes)
- **Macro** — FRED series: CPI, unemployment, Fed funds rate, 10Y yield, M2

Data comes from `yfinance` (markets) and `fredapi` / `pandas-datareader` (macro). Both are free; FRED requires a no-cost API key.

## Sub-modules

| # | Module | Builds on |
|---|--------|-----------|
| [5.1](module-5.1-financial-timeseries-foundations/) | Foundations of Financial Time Series | L1.4, L2.1 |
| [5.2](module-5.2-classical-forecasting/) | Classical Statistical Forecasting (ARIMA, ETS, Prophet) | L1.3, L2.1 |
| [5.3](module-5.3-ml-forecasting/) | ML Forecasting with Tabular Models (XGBoost, LightGBM) | L2.2, L2.3 |
| [5.4](module-5.4-deep-learning-forecasting/) | Deep Learning for Time Series (LSTM, N-BEATS, N-HiTS, PatchTST, TFT) | L3.1, L3.3 |
| [5.5](module-5.5-foundation-models/) | Time-Series Foundation Models (Chronos, TimesFM, TimeGPT, Moirai) | L3.3, L4.3 |
| [5.6](module-5.6-llm-augmented-forecasting/) | LLM-Augmented Forecasting (sentiment, RAG, agents) | L4.1, L4.2, L4.4 |
| [5.7](module-5.7-probabilistic-backtesting/) | Probabilistic Forecasting & Backtesting | L2.1, L4.6 |
| [5.8](module-5.8-capstone/) | Capstone Project — full ensembled forecaster, served | L4.5 |

## Capstone deliverable

A single repo with:
- A pipeline that downloads data, trains 5 model families, and ensembles them under a conformal interval.
- A walk-forward backtest with classical (MAPE/SMAPE/MASE) and trading (Sharpe, max drawdown) metrics.
- A FastAPI endpoint that returns point + interval forecasts plus a Claude-written narrative.
- A Streamlit dashboard for exploring forecasts across tickers.
- MLflow tracking, Docker container, GitHub Actions CI.

## Disclaimer

This module is **educational**. Nothing here is investment advice. Backtested results do not predict future returns. Real money decisions require infrastructure, risk management, and domain expertise far beyond a learning project.

## Suggested pace

Two weeks per sub-module on a part-time schedule, with the capstone (5.8) running in parallel from week 4 onward as you accumulate components.
