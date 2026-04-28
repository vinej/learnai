# Module 5.4 — Deep Learning for Time Series

**Level:** 5 — Extra
**Estimated time:** 2 weeks
**Builds on:** L3.1 (DL foundations), L3.3 (sequence models), 5.1, 5.3

## Goal

Build deep models for forecasting and understand when (and why) they beat boosted trees. Cover the architectures that defined 2020-2025: LSTM, N-BEATS, N-HiTS, PatchTST, Temporal Fusion Transformer.

## Topics

- LSTM/GRU baselines: rolling window dataset, training loop in PyTorch
- Encoder-only Transformers for time series and the "channel-independence" trick
- N-BEATS / N-HiTS: pure feed-forward residual blocks, interpretable and fast
- PatchTST (2023): patching + transformer; one of the strongest deep TS baselines
- Temporal Fusion Transformer (TFT): multi-horizon, mixed exogenous, attention interpretability
- Loss functions: MSE, MAE, Huber, MASE, quantile loss
- The `darts` and `neuralforecast` libraries — both wrap these architectures

## When DL beats trees (and when it doesn't)

| Situation | Winner |
|-----------|--------|
| Few series (1-5), short history (<5y) | Trees / classical |
| Many similar series, panel data | DL (TFT, PatchTST) |
| Need long-horizon multi-step | DL (PatchTST, N-HiTS) |
| Need interpretability + interval forecasts | TFT or trees+conformal |
| Tight latency budget | Trees / ETS |

## Files

| File | What it covers |
|------|----------------|
| `01_dataset_window.py` | `WindowDataset`: produce (X_seq, y) from a time series |
| `02_lstm_pytorch.py` | LSTM baseline written from scratch in PyTorch |
| `03_nbeats_neuralforecast.py` | N-BEATS via `neuralforecast` |
| `04_nhits_neuralforecast.py` | N-HiTS for long-horizon forecasting |
| `05_patchtst.py` | PatchTST, the modern transformer baseline |
| `06_tft_darts.py` | Temporal Fusion Transformer with `darts` and exogenous regressors |
| `07_quantile_loss.py` | Pinball loss for direct interval forecasts |
| `08_compare_to_lgb.py` | Same data, same splits — DL vs LGB head-to-head |

## Exercises

1. Implement a `WindowDataset` and feed it through an LSTM that predicts SPY 5-day returns.
2. Train N-HiTS on monthly CPI YoY 12 months ahead and beat ETS.
3. Train PatchTST on a panel of 8 ETFs simultaneously; compare to per-ticker LGB.
4. Train a TFT with sector relatives and macro regressors; inspect attention weights.
5. Reproduce the comparison in `08_compare_to_lgb.py` for a different target (BTC vol, EUR/USD return, CPI YoY).

## Resources

- `darts`: https://unit8co.github.io/darts/
- `neuralforecast`: https://nixtlaverse.nixtla.io/neuralforecast/
- PatchTST paper: Nie et al. 2023 — "A Time Series is Worth 64 Words"
- N-BEATS paper: Oreshkin et al. 2020
- N-HiTS paper: Challu et al. 2023
- TFT paper: Lim et al. 2021

## Checkpoint

You can: prepare a windowed dataset, train one of {LSTM, N-BEATS, N-HiTS, PatchTST, TFT} to convergence, and articulate which architecture you'd choose for a given dataset and why.
