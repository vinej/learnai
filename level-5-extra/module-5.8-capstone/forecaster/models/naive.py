"""Naive forecaster: y_hat_{t+h} = y_t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ForecastResult


class NaiveForecaster(Forecaster):
    name = "naive"

    def __init__(self) -> None:
        self._last: float | None = None

    def fit(self, history: pd.Series) -> None:
        self._last = float(history.iloc[-1])

    def predict(self, horizon: int) -> ForecastResult:
        assert self._last is not None
        point = np.full(horizon, self._last)
        return ForecastResult(point=point, p10=None, p90=None, samples=None, name=self.name)
