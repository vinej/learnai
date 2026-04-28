"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    ticker: str = Field(..., examples=["SPY"])
    horizon: int = Field(5, ge=1, le=63)
    include_narrative: bool = True


class ModelContribution(BaseModel):
    model: str
    point: float
    weight: float | None = None


class ForecastResponse(BaseModel):
    ticker: str
    as_of: datetime
    horizon: int
    point_forecast: list[float]
    p10: list[float] | None = None
    p90: list[float] | None = None
    coverage_target: float = 0.8
    contributions: list[ModelContribution]
    narrative: str | None = None
