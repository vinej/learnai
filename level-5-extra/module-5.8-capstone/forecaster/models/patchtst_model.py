"""PatchTST wrapper via neuralforecast.

Optional — requires the `[deep]` extra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ForecastResult


class PatchTSTForecaster(Forecaster):
    name = "patchtst"

    def __init__(self, horizon: int = 5, input_size: int = 128, max_steps: int = 200):
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self._nf = None

    def fit(self, history: pd.Series) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import PatchTST

        df = history.reset_index()
        df.columns = ["ds", "y"]
        df["unique_id"] = "series"

        self._nf = NeuralForecast(
            models=[PatchTST(
                h=self.horizon, input_size=self.input_size,
                patch_len=16, stride=8, n_heads=4, encoder_layers=3,
                hidden_size=128, linear_hidden_size=256,
                loss=MAE(), max_steps=self.max_steps,
            )],
            freq=pd.infer_freq(history.index) or "B",
        )
        self._nf.fit(df)

    def predict(self, horizon: int) -> ForecastResult:
        assert self._nf is not None
        preds = self._nf.predict()
        point = preds["PatchTST"].values[:horizon]
        return ForecastResult(point=point, p10=None, p90=None, samples=None, name=self.name)
