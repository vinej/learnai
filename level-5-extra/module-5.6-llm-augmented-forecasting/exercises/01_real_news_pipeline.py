"""
Exercise 1 — Real news pipeline + walk-forward validation.

Build a small daily news-sentiment pipeline:
- Pull headlines for SPY/SPX from a free RSS source (Yahoo Finance, MarketWatch).
  Use feedparser; cache the raw items to disk.
- Score with Claude Haiku (or GPT-4o-mini); aggregate to a daily score.
- Lag by 1 day and add to your LightGBM feature set from 5.3.
- Walk-forward CV: 5 folds, 1-quarter test windows.
- Report mean MAE and direction accuracy with vs without sentiment.

Honest expectation: the gain on a single ticker is small (~2-5 bps MAE).
Real production pipelines combine many feeds and filter by relevance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
