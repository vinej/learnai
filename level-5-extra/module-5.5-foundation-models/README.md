# Module 5.5 — Time-Series Foundation Models (2026)

**Level:** 5 — Extra
**Estimated time:** 1 week
**Builds on:** L3.3 (sequence models), L4.3 (fine-tuning), 5.1, 5.4

## Goal

Use pretrained foundation models for time series — Chronos, TimesFM, TimeGPT, Moirai, Lag-Llama — and learn when zero/few-shot prompting beats training a custom model.

## Why this exists

In 2024-2025 a wave of "TS foundation models" appeared, pretrained on hundreds of billions of timesteps from diverse sources. By 2026, zero-shot forecasts from these models are competitive with — and sometimes beat — bespoke models trained on a single series. The shift mirrors what LLMs did to NLP between 2018 and 2022.

## Key models covered

| Model | Released | Source | Access |
|-------|----------|--------|--------|
| **Chronos** | Mar 2024 | Amazon Science | Open weights (HuggingFace) |
| **Chronos-Bolt** | Nov 2024 | Amazon Science | Open weights, ~250x faster than v1 |
| **TimesFM** | May 2024, v2 Oct 2024 | Google Research | Open weights |
| **TimeGPT-1 / TimeGPT-2** | 2023 / 2024 | Nixtla | Paid API |
| **Moirai** / **Moirai-MoE** | 2024 | Salesforce | Open weights |
| **Lag-Llama** | 2024 | ServiceNow + Mila | Open weights |

By 2026 the field consolidates: expect Chronos-Bolt and TimesFM-v2 to be the production defaults; TimeGPT remains the strongest paid option for long horizons.

## Topics

- Zero-shot forecasting: feed in a series, get back a forecast
- Few-shot fine-tuning with LoRA-style adapters
- When foundation models help vs hurt (small-N, noisy series → big help; high-frequency, asset-specific → mixed)
- Probabilistic forecasts from sample paths
- Cost / latency / context-length tradeoffs

## Files

| File | What it covers |
|------|----------------|
| `01_chronos_zero_shot.py` | Zero-shot SPY return forecast with Chronos-Bolt-Base |
| `02_chronos_probabilistic.py` | Sampling many paths to get distributional forecasts |
| `03_timesfm.py` | TimesFM-v2 zero-shot on FRED CPI YoY |
| `04_timegpt_api.py` | TimeGPT API client (cross-frequency, exogenous) |
| `05_moirai.py` | Moirai / Moirai-MoE on a panel of ETFs |
| `06_lag_llama.py` | Lag-Llama for long-context probabilistic forecasting |
| `07_fine_tune_chronos.py` | LoRA fine-tune Chronos on financial data |
| `08_when_foundation_wins.py` | A/B vs a tuned LightGBM / N-HiTS across asset classes |

## Exercises

1. Run all five foundation models zero-shot on SPY and report MAE on the same 1-year holdout.
2. Sample 500 paths from Chronos and use them to estimate the 10/50/90 quantiles; compare to LGB quantile (5.3).
3. Fine-tune Chronos on 10 years of crypto data; check whether it generalizes to a different coin.
4. Use TimeGPT with `add_history=True` and exogenous vars (DGS10, VIX); compare to its purely-univariate output.
5. Build an "ensemble of foundations": average Chronos, TimesFM, Moirai forecasts and check if it's better than any one alone.

## Resources

- Chronos: https://github.com/amazon-science/chronos-forecasting
- TimesFM: https://github.com/google-research/timesfm
- Moirai: https://github.com/SalesforceAIResearch/uni2ts
- Lag-Llama: https://github.com/time-series-foundation-models/lag-llama
- TimeGPT docs: https://docs.nixtla.io/

## Checkpoint

You can: load a pretrained TS foundation model, produce zero-shot probabilistic forecasts on any series, and articulate when fine-tuning is worth the GPU cost.
