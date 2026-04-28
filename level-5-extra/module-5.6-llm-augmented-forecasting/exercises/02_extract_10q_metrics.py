"""
Exercise 2 — Extract metrics from real 10-Q filings.

Use sec-edgar-downloader to pull the last 5 10-Qs for AAPL.
Run the structured-extraction tool from 03 on each.
Build a panel: rows = filings, columns = (metric, value, period, yoy_change_pct).

Validate:
- Cross-check at least 3 metrics manually against the filing text.
- Plot revenue and EPS over the 5 quarters.
- Discuss any extraction errors and what would prevent them in
  production (better prompts? better tool schema? human review loop?).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
