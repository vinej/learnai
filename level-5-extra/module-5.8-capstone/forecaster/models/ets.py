"""ETS via statsmodels."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecaster.models.base import Forecaster, ForecastResult

warnings.filterwarnings("ignore", category=FutureWarning)


class ETSForecaster(Forecaster):
    name = "ets"

    def __init__(self, trend: str | None = "add", seasonal: str | None = None,
                  seasonal_periods: int | None = None) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fit = None

    def fit(self, history: pd.Series) -> None:
        self._fit = ExponentialSmoothing(
            history,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit()

    def predict(self, horizon: int) -> ForecastResult:
        assert self._fit is not None
        point = np.asarray(self._fit.forecast(horizon).values)
        # Use simulated paths for intervals
        try:
            sims = self._fit.simulate(horizon, repetitions=200, anchor="end").T  # (200, H)
            samples = np.asarray(sims)
            p10 = np.quantile(samples, 0.1, axis=0)
            p90 = np.quantile(samples, 0.9, axis=0)
        except Exception:                      # noqa: BLE001
            samples, p10, p90 = None, None, None
        return ForecastResult(point=point, p10=p10, p90=p90, samples=samples, name=self.name)
