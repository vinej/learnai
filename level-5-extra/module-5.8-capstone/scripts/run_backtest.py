"""Run a walk-forward backtest for a model + ticker, log to MLflow.

Usage:
    python scripts/run_backtest.py --ticker SPY --horizon 5 --model lgb
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np

from forecaster.config import MLRUNS
from forecaster.data.loaders import fetch_market
from forecaster.data.splits import WalkForwardConfig
from forecaster.eval.backtest import backtest
from forecaster.models.ets import ETSForecaster
from forecaster.models.lightgbm_models import LightGBMForecaster
from forecaster.models.naive import NaiveForecaster

MODELS = {
    "naive": NaiveForecaster,
    "ets":   lambda: ETSForecaster(trend=None, seasonal=None),
    "lgb":   lambda: LightGBMForecaster(horizon=5, n_round=300),
}


def main(ticker: str, horizon: int, model_key: str) -> None:
    mlflow.set_tracking_uri(f"file:{MLRUNS}")
    mlflow.set_experiment("forecaster")

    px = fetch_market(ticker)["Close"].asfreq("B").ffill()
    logret = np.log(px / px.shift(1)).dropna()

    cfg = WalkForwardConfig(initial_train=252 * 5, horizon=horizon, step=63)
    factory = MODELS[model_key]

    with mlflow.start_run(run_name=f"{ticker}-{model_key}-h{horizon}"):
        mlflow.log_params({"ticker": ticker, "horizon": horizon, "model": model_key,
                            "initial_train": cfg.initial_train, "step": cfg.step})
        report = backtest(factory, logret, cfg=cfg)
        summary = report.summary()
        mlflow.log_metrics(summary)
        out = report.df
        mlflow.log_text(out.to_csv(index=False), "fold_results.csv")
        print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--model", choices=list(MODELS.keys()), default="lgb")
    args = p.parse_args()
    main(args.ticker, args.horizon, args.model)
