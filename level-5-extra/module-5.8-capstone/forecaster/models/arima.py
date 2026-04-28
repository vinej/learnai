"""SARIMA wrapper."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecaster.models.base import Forecaster, ForecastResult

warnings.filterwarnings("ignore")


class ARIMAForecaster(Forecaster):
    name = "arima"

    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fit = None

    def fit(self, history: pd.Series) -> None:
        self._fit = ARIMA(history, order=self.order, seasonal_order=self.seasonal_order).fit()

    def predict(self, horizon: int) -> ForecastResult:
        assert self._fit is not None
        fc_obj = self._fit.get_forecast(horizon)
        point = np.asarray(fc_obj.predicted_mean.values)
        ci = fc_obj.conf_int(alpha=0.2)         # 80% PI
        p10 = ci.iloc[:, 0].values
        p90 = ci.iloc[:, 1].values
        return ForecastResult(point=point, p10=p10, p90=p90, samples=None, name=self.name)
