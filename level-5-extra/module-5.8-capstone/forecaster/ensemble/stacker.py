"""Stacking ensemble: a Ridge meta-learner over per-model point forecasts.

The stacker is fit on a HELD-OUT validation window (NEVER the training
window of the base models, NEVER the test window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Stacker:
    def __init__(self, alpha: float = 1.0):
        self.meta = Ridge(alpha=alpha, fit_intercept=False, positive=True)
        self.model_names: list[str] = []

    def fit(self, val_preds: pd.DataFrame, val_actual: np.ndarray) -> None:
        """val_preds columns = model names; index aligned with val_actual."""
        self.model_names = list(val_preds.columns)
        self.meta.fit(val_preds.values, val_actual)

    def predict(self, test_preds: pd.DataFrame) -> np.ndarray:
        return self.meta.predict(test_preds[self.model_names].values)

    @property
    def weights(self) -> dict[str, float]:
        return {n: float(w) for n, w in zip(self.model_names, self.meta.coef_)}
