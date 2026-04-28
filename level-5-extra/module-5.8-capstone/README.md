# Module 5.8 — Capstone Project

**Level:** 5 — Extra
**Estimated time:** 2-3 weeks
**Builds on:** ALL of 5.1-5.7, plus L4.5 (MLOps & deployment)

## Goal

Tie everything together. Build, evaluate, deploy, and monitor a complete forecasting service. The deliverable is a single repo that someone could clone and run end-to-end.

## What you build

A self-contained Python package `forecaster/` with:

- **Data layer** — `yfinance` + FRED loaders with on-disk Parquet cache, point-in-time correctness, walk-forward iterators.
- **Model zoo** — pluggable forecasters with a common interface:
  - `NaiveForecaster`, `ETSForecaster`, `ARIMAForecaster`
  - `LightGBMForecaster` (point + quantile)
  - `PatchTSTForecaster` (deep)
  - `ChronosForecaster` (foundation, zero-shot)
  - `LLMSentimentScorer` (Claude-based feature)
- **Ensemble** — stacking ensemble with conformal calibration on top.
- **Evaluation** — walk-forward backtest with classical (MAE, MASE, CRPS) and trading (Sharpe, drawdown) metrics.
- **Tracking** — MLflow logs every run.
- **Serving** — FastAPI endpoint returning point + interval + LLM narrative.
- **Dashboard** — Streamlit UI for exploring forecasts and backtests.
- **Containerization** — Dockerfile, GitHub Actions CI, basic monitoring.

## Folder layout

```
module-5.8-capstone/
  README.md                  ← this file
  INSTALL.md
  Dockerfile
  pyproject.toml
  forecaster/
    __init__.py
    config.py
    data/
      loaders.py
      splits.py
    models/
      base.py                ← Forecaster ABC
      naive.py
      ets.py
      arima.py
      lightgbm_models.py
      patchtst_model.py
      chronos_model.py
      llm_sentiment.py
    ensemble/
      stacker.py
      conformal.py
    eval/
      metrics.py
      backtest.py
    serving/
      api.py                 ← FastAPI app
      schemas.py
    narrate/
      narrator.py            ← Claude-based explainer
  app/
    streamlit_app.py
  tests/
    test_models.py
    test_ensemble.py
    test_api.py
  scripts/
    run_backtest.py
    train_all.py
    serve.py
  .github/
    workflows/
      ci.yml
```

## Milestones (suggested ordering)

1. **Week 1 — Skeleton + data + first model**
   - `forecaster.data.loaders` (port from 5.1).
   - `forecaster.models.base.Forecaster` ABC.
   - `NaiveForecaster`, `ETSForecaster`.
   - First walk-forward backtest on SPY 5-day return.

2. **Week 1-2 — Add ML and DL models**
   - `LightGBMForecaster` (point + quantile).
   - `PatchTSTForecaster` via `neuralforecast`.
   - Run multi-model backtest, log all to MLflow.

3. **Week 2 — Foundation model + LLM features**
   - `ChronosForecaster` (zero-shot).
   - `LLMSentimentScorer` from a free RSS feed.
   - Walk-forward eval with sentiment as a feature.

4. **Week 2-3 — Ensemble + conformal**
   - Stacking on a held-out validation window.
   - Conformal / CQR calibration on top.
   - Final report: forecasting + trading metrics, regime breakdown.

5. **Week 3 — Serving and shipping**
   - FastAPI endpoint with pydantic schemas.
   - Streamlit dashboard.
   - Dockerfile, GitHub Actions CI (lint + tests).
   - README + how-to.

## Definition of done

- `pytest -q` passes.
- `python scripts/run_backtest.py --ticker SPY --horizon 5` produces a backtest report.
- `python scripts/serve.py` starts the API; `curl localhost:8000/forecast?ticker=SPY&horizon=5` returns a valid response with a point forecast, an 80% PI, per-model contributions, and a Claude-generated narrative.
- `streamlit run app/streamlit_app.py` opens a working dashboard.
- `docker build -t forecaster . && docker run -p 8000:8000 forecaster` works.
- The CI workflow runs on PR and is green.

## What this is NOT

- Not a trading system — it's a forecasting service. Trading metrics are reported as honesty checks, not as an instruction to deploy capital.
- Not financial advice. See top-level README.

## Files in this module

The folder ships with stub starter files for each of the components above, with TODOs and architectural hints. You build the rest.
