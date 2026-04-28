"""
04 — Prophet

Prophet (Meta) decomposes a series into:
    y(t) = trend(t) + seasonality(t) + holidays(t) + epsilon

It handles missing data, irregular spacing, multiple seasonalities,
and lets you inject external regressors. Best on series with strong
human-calendar effects (retail, web traffic, support tickets).

For finance it's a fine baseline on lower-frequency macro/asset series
but won't capture volatility clustering or fat tails.

Run: python 04_prophet_basics.py
"""
from __future__ import annotations

import logging

import pandas as pd

from _common import fetch_market

# Quiet Prophet's logger
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


if __name__ == "__main__":
    try:
        from prophet import Prophet
    except ImportError:
        print("Install prophet first: pip install prophet")
        raise SystemExit(0)

    spy = fetch_market("SPY")["Close"].asfreq("B").ffill()

    # Prophet expects columns named ds (date) and y (value)
    df = spy.reset_index()
    df.columns = ["ds", "y"]

    train = df.iloc[:-252]   # all but last year
    test = df.iloc[-252:]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,    # markets are M-F so weekly is degenerate
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=252, freq="B", include_history=False)
    fc = m.predict(future)
    fc = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]

    aligned = fc.join(test.set_index("ds")["y"], how="inner")
    err = (aligned["y"] - aligned["yhat"]).abs()
    print(f"Prophet MAE on SPY 1Y holdout: ${err.mean():.2f}")
    print(f"Mean width of 80% CI:           ${(aligned['yhat_upper'] - aligned['yhat_lower']).mean():.2f}")
    print("\nLesson: Prophet's intervals widen with horizon (good), but on")
    print("equity prices the trend assumption is not really right. Modeling")
    print("returns instead of price would be more honest.")
