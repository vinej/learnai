"""LightGBM forecaster — point + quantile.

Uses direct (per-horizon) quantile regression for intervals.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ForecastResult


def _build_features(s: pd.Series) -> pd.DataFrame:
    """Lag and rolling features. All shifted to avoid look-ahead."""
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for k in (1, 2, 3, 5, 10, 21):
        df[f"lag{k}"] = df["y"].shift(k)
    base = df["y"].shift(1)
    for w in (5, 10, 21, 63):
        df[f"rmean{w}"] = base.rolling(w).mean()
        df[f"rstd{w}"] = base.rolling(w).std()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


class LightGBMForecaster(Forecaster):
    name = "lightgbm"

    def __init__(self, horizon: int = 5, n_round: int = 300):
        self.horizon = horizon
        self.n_round = n_round
        self._models: dict[str, list[lgb.Booster]] = {"point": [], "p10": [], "p90": []}
        self._feature_cols: list[str] = []
        self._last_features: pd.Series | None = None

    def _prep(self, s: pd.Series) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
        df = _build_features(s)
        # Direct multi-horizon: y_target_h = y[t + h]
        targets = {h: df["y"].shift(-h).values for h in range(1, self.horizon + 1)}
        df = df.dropna()
        X = df.drop(columns=["y"])
        targets = {h: y[df.index.map(lambda i: i in df.index)] for h, y in targets.items()}
        # Trim NaN tails per horizon
        return X, {h: targets[h][:len(X)] for h in targets}

    def fit(self, history: pd.Series) -> None:
        X, ys = self._prep(history)
        self._feature_cols = list(X.columns)
        for h in range(1, self.horizon + 1):
            mask = ~np.isnan(ys[h])
            X_h, y_h = X.iloc[mask], ys[h][mask]
            self._models["point"].append(self._fit(X_h, y_h, "regression_l1"))
            self._models["p10"].append(self._fit(X_h, y_h, "quantile", alpha=0.1))
            self._models["p90"].append(self._fit(X_h, y_h, "quantile", alpha=0.9))
        # Last available feature row (most recent ready-to-predict)
        self._last_features = _build_features(history).dropna().iloc[-1][self._feature_cols]

    def _fit(self, X, y, objective: str, alpha: float | None = None) -> lgb.Booster:
        params = {"objective": objective, "learning_rate": 0.03,
                   "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1}
        if alpha is not None:
            params["alpha"] = alpha
        return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=self.n_round)

    def predict(self, horizon: int) -> ForecastResult:
        assert horizon <= self.horizon
        assert self._last_features is not None
        X = self._last_features.values.reshape(1, -1)
        point = np.array([self._models["point"][h - 1].predict(X)[0] for h in range(1, horizon + 1)])
        p10 = np.array([self._models["p10"][h - 1].predict(X)[0] for h in range(1, horizon + 1)])
        p90 = np.array([self._models["p90"][h - 1].predict(X)[0] for h in range(1, horizon + 1)])
        return ForecastResult(point=point, p10=p10, p90=p90, samples=None, name=self.name)
