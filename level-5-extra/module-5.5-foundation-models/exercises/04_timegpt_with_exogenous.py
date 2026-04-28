"""
Exercise 4 — TimeGPT with exogenous regressors.

Forecast SPY 21 days ahead. Three configurations:
A) Univariate
B) + DGS10 (10Y yield) as exogenous
C) + DGS10 + ^VIX as exogenous

Run each on the SAME walk-forward windows (5 windows of 21 days).
Report mean MAE and rank.

Discuss: when do exogenous variables help? When do they hurt?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement using nixtla.NixtlaClient.forecast(... X_df=...)
