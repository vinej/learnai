# Setup — Module 5.5

```bash
pip install chronos-forecasting           # Amazon Chronos / Chronos-Bolt
pip install nixtla                        # TimeGPT API client (paid)
pip install uni2ts                        # Salesforce Moirai
# Lag-Llama requires installing from GitHub:
pip install git+https://github.com/time-series-foundation-models/lag-llama.git

# TimesFM requires its own install — see Google's repo:
# https://github.com/google-research/timesfm
# (As of 2026 the cleanest path is the HuggingFace adapters in `timesfm-pytorch`.)
pip install timesfm
```

## API keys

For TimeGPT (Nixtla) you need a paid key — sign up at https://nixtla.io/.
Set in `level-5-extra/.env`:

```env
NIXTLA_API_KEY=...
```

All other models in this module run locally (CPU works for the small
ones; GPU recommended for Lag-Llama and Moirai-MoE).

## Model weights

The first run of each example downloads weights to `~/.cache/huggingface/`.
Sizes (approx, late 2025 catalogues):

- Chronos-Bolt-Base: ~200 MB
- Chronos-Bolt-Small: ~50 MB
- TimesFM-v2 (200M params): ~800 MB
- Moirai-base: ~100 MB
- Moirai-MoE-base: ~300 MB
- Lag-Llama: ~300 MB
