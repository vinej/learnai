"""Walk-forward backtest harness — fits models per fold, collects metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecaster.data.splits import WalkForwardConfig, walk_forward
from forecaster.eval.metrics import coverage, mae, rmse


@dataclass
class FoldResult:
    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    point_mae: float
    point_rmse: float
    cov_80: float | None
    width_80: float | None


@dataclass
class BacktestReport:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])

    def summary(self) -> dict:
        d = self.df
        return {
            "n_folds": int(len(d)),
            "mae_mean": float(d["point_mae"].mean()),
            "mae_std": float(d["point_mae"].std()),
            "rmse_mean": float(d["point_rmse"].mean()),
            "cov_80_mean": float(d["cov_80"].dropna().mean()) if d["cov_80"].notna().any() else float("nan"),
            "width_80_mean": float(d["width_80"].dropna().mean()) if d["width_80"].notna().any() else float("nan"),
        }


def backtest(model_factory, series: pd.Series, *, cfg: WalkForwardConfig
              ) -> BacktestReport:
    """model_factory: () -> Forecaster   (called per fold to refit)."""
    df = pd.DataFrame({"y": series.values}, index=series.index)
    report = BacktestReport()
    for i, (tr, te) in enumerate(walk_forward(df, cfg)):
        model = model_factory()
        model.fit(tr["y"])
        result = model.predict(cfg.horizon)
        actual = te["y"].values[:cfg.horizon]
        m = mae(actual, result.point); r = rmse(actual, result.point)
        cov80 = width80 = None
        if result.p10 is not None and result.p90 is not None:
            cov80 = coverage(actual, result.p10, result.p90)
            width80 = float(np.mean(result.p90 - result.p10))
        report.folds.append(FoldResult(
            fold=i, test_start=te.index[0], test_end=te.index[-1],
            point_mae=m, point_rmse=r, cov_80=cov80, width_80=width80,
        ))
    return report
