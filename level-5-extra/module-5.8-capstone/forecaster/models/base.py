"""Base interface every forecaster implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    point: np.ndarray            # (horizon,)
    p10: np.ndarray | None       # (horizon,) or None
    p90: np.ndarray | None
    samples: np.ndarray | None   # (n_samples, horizon) or None
    name: str


class Forecaster(ABC):
    """Common contract across naive / classical / ML / DL / foundation models.

    Inputs are a univariate target series indexed by date. Subclasses are
    free to use additional features they construct from the index.
    """
    name: str = "Forecaster"

    @abstractmethod
    def fit(self, history: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        ...
