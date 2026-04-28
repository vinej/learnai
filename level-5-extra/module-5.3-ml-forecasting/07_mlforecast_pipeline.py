"""
07 — Tidy ML pipeline with mlforecast.

Nixtla's `mlforecast` wraps the entire feature-engineering + walk-forward
cross-validation + recursive prediction loop. It accepts long-format
DataFrames (unique_id, ds, y) and any sklearn-compatible regressor.

Run: python 07_mlforecast_pipeline.py
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from _common import fetch_market

try:
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from mlforecast.lag_transforms import RollingMean, RollingStd
except ImportError:
    print("Install: pip install mlforecast")
    raise SystemExit(0)


def to_long(prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for ticker, s in prices.items():
        r = np.log(s / s.shift(1)).dropna()
        rows.append(pd.DataFrame({
            "unique_id": ticker,
            "ds": r.index,
            "y": r.values,
        }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    prices = {t: fetch_market(t)["Close"] for t in ("SPY", "QQQ", "XLK", "XLF", "XLE")}
    long_df = to_long(prices)

    fcst = MLForecast(
        models={"lgb": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=100, verbose=-1
        )},
        freq="B",
        lags=[1, 2, 5, 10, 21],
        lag_transforms={
            1: [RollingMean(window_size=5), RollingStd(window_size=5)],
            5: [RollingMean(window_size=21)],
        },
        date_features=["dayofweek", "month"],
        target_transforms=[Differences([0])],   # already returns; placeholder
    )
    fcst.fit(long_df, id_col="unique_id", time_col="ds", target_col="y")

    horizon = 5
    preds = fcst.predict(h=horizon)
    print("5-day-ahead forecasts (returns) per ticker:")
    print(preds.pivot(index="ds", columns="unique_id", values="lgb").round(5))

    # Cross-validation on the panel
    cv = fcst.cross_validation(
        df=long_df,
        h=horizon,
        n_windows=4,
        step_size=126,
        id_col="unique_id", time_col="ds", target_col="y",
    )
    cv["abs_err"] = (cv["y"] - cv["lgb"]).abs()
    print("\nMean MAE per ticker across CV folds:")
    print(cv.groupby("unique_id")["abs_err"].mean().round(5))
