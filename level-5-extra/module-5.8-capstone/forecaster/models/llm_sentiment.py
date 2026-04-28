"""LLM-based sentiment scorer — used as a feature, NOT a forecaster.

In the capstone, the LightGBMForecaster takes an optional `sentiment`
column. This module provides a thin wrapper around Anthropic to score
headlines and produce a daily aggregate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from forecaster.config import SETTINGS

SCORER_SYSTEM = """You score financial headlines for short-term (1-5 day) market impact.
Return strict JSON: a list of objects with keys index (int), score (float, -1..+1)."""


@dataclass
class HeadlineItem:
    timestamp: pd.Timestamp
    headline: str


class LLMSentimentScorer:
    def __init__(self, model: str | None = None, batch: int = 20):
        self.model = model or SETTINGS.claude_fast
        self.batch = batch

    def score_batch(self, headlines: list[str]) -> list[float]:
        from anthropic import Anthropic
        client = Anthropic(api_key=SETTINGS.anthropic_key)
        msg = client.messages.create(
            model=self.model, max_tokens=1024,
            system=SCORER_SYSTEM,
            messages=[{"role": "user",
                       "content": json.dumps({"headlines": headlines})}],
        )
        text = msg.content[0].text
        items = json.loads(text[text.find("["):text.rfind("]") + 1])
        scores = [0.0] * len(headlines)
        for item in items:
            i = int(item["index"])
            if 0 <= i < len(headlines):
                scores[i] = float(item["score"])
        return scores

    def daily_aggregate(self, items: list[HeadlineItem]) -> pd.Series:
        """Score all headlines, group by date (US/Eastern), return mean score."""
        if not items:
            return pd.Series(dtype=float)
        all_scores: list[float] = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            all_scores.extend(self.score_batch([h.headline for h in chunk]))
        df = pd.DataFrame({
            "ts": [it.timestamp for it in items],
            "score": all_scores,
        })
        df["date"] = df["ts"].dt.tz_convert("US/Eastern").dt.normalize()
        return df.groupby("date")["score"].mean()
