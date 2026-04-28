"""Streamlit dashboard for exploring forecasts.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from forecaster.data.loaders import fetch_market
from forecaster.models.ets import ETSForecaster
from forecaster.models.lightgbm_models import LightGBMForecaster
from forecaster.models.naive import NaiveForecaster

st.set_page_config(page_title="Forecaster", layout="wide")
st.title("Capstone — multi-model financial forecaster")

ticker = st.sidebar.text_input("Ticker", value="SPY")
horizon = st.sidebar.slider("Horizon (days)", 1, 21, 5)
run = st.sidebar.button("Forecast")

if run:
    df = fetch_market(ticker)
    if df.empty:
        st.error(f"No data for {ticker}")
        st.stop()

    px = df["Close"].asfreq("B").ffill()
    logret = np.log(px / px.shift(1)).dropna()

    st.subheader(f"{ticker} — last 1y price")
    st.line_chart(px.tail(252).rename("close"))

    naive = NaiveForecaster(); naive.fit(logret)
    ets = ETSForecaster(trend=None, seasonal=None); ets.fit(logret)
    lgb_m = LightGBMForecaster(horizon=horizon); lgb_m.fit(logret)

    n = naive.predict(horizon).point
    e = ets.predict(horizon).point
    r = lgb_m.predict(horizon)

    fc = pd.DataFrame({
        "naive": n.cumsum(),
        "ets": e.cumsum(),
        "lgb": r.point.cumsum(),
        "ensemble": ((n + e + r.point) / 3).cumsum(),
    })
    fc.index = [f"+{i + 1}d" for i in range(horizon)]
    st.subheader("Cumulative log-return forecast by model")
    st.dataframe(fc.round(5))
    st.line_chart(fc)

    if r.p10 is not None and r.p90 is not None:
        st.subheader("LightGBM 80% prediction interval (cumulative log return)")
        bands = pd.DataFrame({
            "p10": r.p10.cumsum(),
            "p50": r.point.cumsum(),
            "p90": r.p90.cumsum(),
        }, index=fc.index)
        st.line_chart(bands)
