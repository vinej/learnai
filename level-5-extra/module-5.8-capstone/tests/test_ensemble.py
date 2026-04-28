"""Tests for the stacker and conformal calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.ensemble.conformal import cqr_pad, split_conformal_pad
from forecaster.ensemble.stacker import Stacker


def test_stacker_basic():
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.normal(0, 1, n)
    preds = pd.DataFrame({
        "a": truth + rng.normal(0, 0.5, n),
        "b": truth + rng.normal(0, 0.7, n),
    })
    s = Stacker(alpha=1.0)
    s.fit(preds, truth)
    out = s.predict(preds)
    # Stacked MAE should beat the worst single model.
    mae_a = float(np.abs(preds["a"] - truth).mean())
    mae_b = float(np.abs(preds["b"] - truth).mean())
    mae_stack = float(np.abs(out - truth).mean())
    assert mae_stack <= max(mae_a, mae_b) + 1e-6


def test_split_conformal_pad():
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q = split_conformal_pad(residuals, alpha=0.2)
    assert 0.4 <= q <= 0.5  # ~80th percentile


def test_cqr_pad():
    y = np.array([0.0, 0.5, -0.3, 0.2])
    lo = np.array([-0.4, 0.0, -0.5, 0.0])
    hi = np.array([0.4, 0.6, 0.0, 0.5])
    pad = cqr_pad(lo, hi, y, alpha=0.2)
    assert pad >= 0
