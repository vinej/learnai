"""Chronos-Bolt wrapper — zero-shot, no fit needed (we keep history)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ForecastResult


class ChronosForecaster(Forecaster):
    name = "chronos_bolt"

    def __init__(self, repo_id: str = "amazon/chronos-bolt-base"):
        self.repo_id = repo_id
        self._history: pd.Series | None = None
        self._pipe = None

    def _ensure_pipe(self):
        if self._pipe is None:
            import torch
            from chronos import ChronosBoltPipeline
            self._pipe = ChronosBoltPipeline.from_pretrained(
                self.repo_id,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )

    def fit(self, history: pd.Series) -> None:
        self._history = history.copy()

    def predict(self, horizon: int) -> ForecastResult:
        import torch
        assert self._history is not None
        self._ensure_pipe()
        ctx = torch.tensor(self._history.values, dtype=torch.float32).unsqueeze(0)
        quantiles, means = self._pipe.predict_quantiles(
            context=ctx, prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        qs = quantiles.cpu().numpy()[0]
        return ForecastResult(
            point=means.cpu().numpy().flatten(),
            p10=qs[:, 0], p90=qs[:, 2],
            samples=None, name=self.name,
        )
