"""Sanity tests for the forecaster zoo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.models.ets import ETSForecaster
from forecaster.models.lightgbm_models import LightGBMForecaster
from forecaster.models.naive import NaiveForecaster


@pytest.fixture
def synthetic() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    vals = np.cumsum(rng.normal(0, 0.01, size=len(idx)))
    return pd.Series(vals, index=idx, name="logret")


def test_naive(synthetic):
    m = NaiveForecaster(); m.fit(synthetic)
    out = m.predict(5)
    assert out.point.shape == (5,)
    assert np.allclose(out.point, synthetic.iloc[-1])


def test_ets(synthetic):
    m = ETSForecaster(trend=None, seasonal=None); m.fit(synthetic)
    out = m.predict(5)
    assert out.point.shape == (5,)
    assert np.all(np.isfinite(out.point))


def test_lgb(synthetic):
    m = LightGBMForecaster(horizon=5, n_round=50); m.fit(synthetic)
    out = m.predict(5)
    assert out.point.shape == (5,)
    assert out.p10 is not None and out.p90 is not None
    # Quantiles should bracket point most of the time (not always after fit on small data)
    # We only assert basic sanity here.
    assert np.all(out.p10 <= out.p90 + 1e-6)
