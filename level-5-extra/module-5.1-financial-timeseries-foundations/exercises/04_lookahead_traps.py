"""
Exercise 4 — Find and fix three look-ahead bugs.

The function below "predicts" tomorrow's SPY return using a few simple
features. The reported R^2 is suspiciously high. There are three
look-ahead bugs. Identify them, fix them, and rerun. The honest R^2
should be near zero.

Bugs (don't peek until you've tried):
- Look at how features are computed.
- Look at how the standardization is done.
- Look at the shift on the target.

When you've fixed all three, the R^2 should be < 0.005 on holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import fetch_market


def build_features_BUGGY(px: pd.Series) -> pd.DataFrame:
    r = np.log(px / px.shift(1))
    df = pd.DataFrame({"r": r})

    # BUG #1 (rolling without shift — uses today in today's feature)
    df["mom20"] = r.rolling(20).mean()
    df["vol20"] = r.rolling(20).std()

    # BUG #2 (standardization uses the WHOLE series mean/std)
    for c in ("mom20", "vol20"):
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    # BUG #3 (target shift — predicting today from today)
    df["y"] = r
    return df.dropna()


if __name__ == "__main__":
    px = fetch_market("SPY")["Close"]
    df = build_features_BUGGY(px)

    n = len(df)
    train, test = df.iloc[:int(n * 0.7)], df.iloc[int(n * 0.7):]
    X_cols = ["mom20", "vol20"]
    model = LinearRegression().fit(train[X_cols], train["y"])
    pred = model.predict(test[X_cols])
    r2 = r2_score(test["y"], pred)
    print(f"Reported R^2 (buggy): {r2:.4f}  <-- too good to be true")

    # TODO: write build_features_FIXED that addresses all three bugs.
    # TODO: rerun and assert that R^2 is small and roughly zero-centered.
