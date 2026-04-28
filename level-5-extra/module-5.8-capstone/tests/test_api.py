"""Smoke test for the FastAPI service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forecaster.serving.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_forecast_smoke(client):
    # Skip if no internet / yfinance offline
    pytest.importorskip("yfinance")
    r = client.post("/forecast", json={
        "ticker": "SPY", "horizon": 3, "include_narrative": False,
    })
    if r.status_code == 404:
        pytest.skip("data fetch failed (offline?)")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "SPY"
    assert len(body["point_forecast"]) == 3
    assert len(body["contributions"]) >= 1
