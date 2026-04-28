"""
07 — Fine-tuning Chronos on financial data.

When a foundation model is "almost right" but consistently biased on
your domain (e.g., underestimating crypto vol), a small LoRA fine-tune
on a few thousand series can close the gap without losing the
zero-shot generality.

Run: python 07_fine_tune_chronos.py
"""
from __future__ import annotations

# Fine-tuning Chronos requires the official `scripts/training/` from the
# Amazon repo (https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training).
# The recipe is well-documented; this file walks through the high-level
# steps so you can apply them.

WORKFLOW = """
1. Prepare data
   - Build an Arrow / Parquet dataset of (series_id, timestamps, values).
   - Suggested: 5-10 years of daily returns for SPY, QQQ, IWM, EFA, BTC,
     ETH, EURUSD, USDJPY, DGS10, CPI YoY (~10k series x 1k steps after
     resampling).

2. Configure training
   - Clone amazon-science/chronos-forecasting.
   - Use scripts/training/configs/chronos-bolt-base.yaml as a template.
   - Reduce learning_rate to ~1e-4, set max_steps to 5000-20000 depending
     on data volume, enable LoRA (the repo supports it via peft).

3. Launch
     python -m scripts.training.train \\
         --config configs/chronos-bolt-base-finance.yaml \\
         --output-dir runs/chronos-finance \\
         --resume-from amazon/chronos-bolt-base

4. Evaluate
   - Hold out the last 1-2 years of each series.
   - Run zero-shot Chronos-Bolt-Base AND your fine-tuned checkpoint on
     the same holdout. Compare MAE / pinball loss.

5. Test generalization
   - Try a series you DID NOT train on (e.g., a sector ETF, a different
     coin). The fine-tune should still work zero-shot if you preserved
     diversity in training.

Common pitfalls
- Training on prices instead of returns -> model collapses to "predict
  current value forever".
- Mixing frequencies (daily + monthly) without conditioning -> hurts.
- Too few series -> you're overfitting; classical models will win.
"""

if __name__ == "__main__":
    print(WORKFLOW)
