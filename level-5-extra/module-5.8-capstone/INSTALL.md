# Setup — Module 5.8

The capstone uses everything from 5.1-5.7. Easiest path is the shared
Level 5 environment:

```bash
cd level-5-extra
python -m venv .venv
# activate
pip install -r requirements.txt
```

Then from this folder:

```bash
pip install -e .            # install the forecaster/ package in editable mode
```

(Requires `pyproject.toml` — included as a starter.)

## Run targets

```bash
# Full backtest on SPY, 5-day horizon, all models
python scripts/run_backtest.py --ticker SPY --horizon 5

# Train and pickle models for serving
python scripts/train_all.py --ticker SPY

# Start the API
python scripts/serve.py        # equivalent to: uvicorn forecaster.serving.api:app --reload

# Launch the dashboard
streamlit run app/streamlit_app.py

# Tests + lint
pytest -q
ruff check .
```

## Docker

```bash
docker build -t forecaster .
docker run -p 8000:8000 --env-file ../.env forecaster
```

## API keys

Same `.env` as the rest of Level 5:

```env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
FRED_API_KEY=...
NIXTLA_API_KEY=...   # optional
```
