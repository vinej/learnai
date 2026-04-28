"""FastAPI service exposing the forecaster.

Run:
    uvicorn forecaster.serving.api:app --reload
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from forecaster.data.loaders import fetch_market
from forecaster.models.ets import ETSForecaster
from forecaster.models.lightgbm_models import LightGBMForecaster
from forecaster.models.naive import NaiveForecaster
from forecaster.narrate.narrator import narrate
from forecaster.serving.schemas import (
    ForecastRequest, ForecastResponse, ModelContribution,
)

app = FastAPI(title="Forecaster", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        df = fetch_market(req.ticker)
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(404, f"Could not fetch {req.ticker}: {e}")
    if df.empty:
        raise HTTPException(404, f"No data for {req.ticker}")

    # Forecast the LOG RETURN, then convert to price-level point forecast.
    import numpy as np
    px = df["Close"].asfreq("B").ffill()
    logret = np.log(px / px.shift(1)).dropna()

    # Fit a tiny zoo (the full capstone uses more)
    naive = NaiveForecaster(); naive.fit(logret)
    ets = ETSForecaster(trend=None, seasonal=None); ets.fit(logret)
    lgb_m = LightGBMForecaster(horizon=req.horizon); lgb_m.fit(logret)

    naive_p = naive.predict(req.horizon).point
    ets_p = ets.predict(req.horizon).point
    lgb_r = lgb_m.predict(req.horizon)

    # Simple equal-weight ensemble of point forecasts
    point_logret = (naive_p + ets_p + lgb_r.point) / 3
    p10 = lgb_r.p10
    p90 = lgb_r.p90

    last_px = float(px.iloc[-1])
    point_levels = [float(last_px * np.exp(np.cumsum(point_logret)[i])) for i in range(req.horizon)]
    p10_levels = ([float(last_px * np.exp(np.cumsum(p10)[i])) for i in range(req.horizon)]
                   if p10 is not None else None)
    p90_levels = ([float(last_px * np.exp(np.cumsum(p90)[i])) for i in range(req.horizon)]
                   if p90 is not None else None)

    contributions = [
        ModelContribution(model="naive", point=float(naive_p[-1]), weight=1 / 3),
        ModelContribution(model="ets", point=float(ets_p[-1]), weight=1 / 3),
        ModelContribution(model="lightgbm", point=float(lgb_r.point[-1]), weight=1 / 3),
    ]

    narrative = None
    if req.include_narrative:
        try:
            narrative = narrate({
                "ticker": req.ticker,
                "horizon_days": req.horizon,
                "as_of": datetime.now(tz=timezone.utc).date().isoformat(),
                "point_forecast_pct": float(point_logret.sum() * 100),
                "interval_pct": ([float(p10.sum() * 100), float(p90.sum() * 100)]
                                  if p10 is not None and p90 is not None else None),
                "interval_level": 0.8,
                "model_contributions_pct": {
                    "naive": float(naive_p[-1] * 100),
                    "ets": float(ets_p[-1] * 100),
                    "lightgbm": float(lgb_r.point[-1] * 100),
                },
            })
        except Exception:                        # noqa: BLE001
            narrative = None

    return ForecastResponse(
        ticker=req.ticker,
        as_of=datetime.now(tz=timezone.utc),
        horizon=req.horizon,
        point_forecast=point_levels,
        p10=p10_levels, p90=p90_levels,
        contributions=contributions,
        narrative=narrative,
    )
