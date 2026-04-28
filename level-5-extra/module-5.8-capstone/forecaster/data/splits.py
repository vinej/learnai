"""Walk-forward iterators for honest model evaluation."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardConfig:
    initial_train: int
    horizon: int
    step: int
    mode: str = "anchored"     # "anchored" or "rolling"


def walk_forward(df: pd.DataFrame, cfg: WalkForwardConfig
                  ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    end = cfg.initial_train
    while end + cfg.horizon <= n:
        if cfg.mode == "anchored":
            tr = df.iloc[:end]
        elif cfg.mode == "rolling":
            tr = df.iloc[max(0, end - cfg.initial_train):end]
        else:
            raise ValueError(cfg.mode)
        te = df.iloc[end:end + cfg.horizon]
        yield tr, te
        end += cfg.step
