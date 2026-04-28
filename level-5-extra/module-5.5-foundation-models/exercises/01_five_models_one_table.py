"""
Exercise 1 — Five foundation models, one comparison table.

For SPY 21-day-ahead price forecast:
- Chronos-Bolt-Base
- TimesFM-2.0-200m
- Moirai-1.1-R-base
- TimeGPT-1 (if you have an API key)
- Lag-Llama (open weights)

Same chronological holdout (last calendar year).
Output a Markdown table:
| model | MAE | RMSE | 80% PI coverage | inference time (s) |

Discuss the result in 5-8 lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement; skip TimeGPT if no API key.
