"""
Exercise 5 — Measure prompt-caching savings on a real workload.

Take your real-news pipeline from exercise 1. Run it for 100 headlines
under three configurations:

A) No caching, vanilla calls.
B) Cache the system prompt only.
C) Cache the system prompt AND a "rolling 24h news context" block that
   you pass on every call.

For each:
- Total wall time
- Total input tokens / cache_creation / cache_read tokens
- Estimated cost (use Anthropic's published pricing for the model you choose)

Discuss: what's the breakeven point for caching to pay off?
(Hint: Anthropic's caches expire ~5 min by default; longer with the
1-hour cache option.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement and write the conclusion.
