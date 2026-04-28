"""
Exercise 3 — Fine-tune Chronos on BTC, evaluate on ETH (zero-shot).

Workflow:
1. Build a training set of 10 years of BTC daily log returns (one series).
2. LoRA fine-tune Chronos-Bolt-Small for 2000 steps.
3. Evaluate the fine-tuned and base models on:
   - BTC holdout (in-domain)
   - ETH zero-shot (out-of-domain, no ETH in training)

Hypothesis: fine-tune improves BTC, slightly hurts ETH vs base. If the
training set is too narrow, the fine-tune overfits to BTC-specific
volatility patterns.

Bonus: extend training to (BTC, ETH, SOL) and re-evaluate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: see 07_fine_tune_chronos.py for the workflow; this is a multi-hour
# exercise with GPU recommended.
