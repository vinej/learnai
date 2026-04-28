# Module 5.6 — LLM-Augmented Forecasting

**Level:** 5 — Extra
**Estimated time:** 1-2 weeks
**Builds on:** L4.1 (LLMs), L4.2 (RAG), L4.4 (agents), 5.3, 5.5

## Goal

Use LLMs to build features and narratives that traditional forecasters can't access: news sentiment, earnings call tone, scenario analysis, agentic research loops, and forecast explanations. The model still does the numbers — the LLM contributes signals from text and structured reasoning.

## What LLMs are good at (and not) for forecasting

**Good**
- Extracting structured signals from unstructured text (sentiment, events, regime classification).
- Generating plausible scenarios.
- Summarizing forecasts and uncertainty into a readable narrative.
- Tool-calling agents that orchestrate other forecasters.

**Not good**
- Producing point or quantile forecasts directly from raw price history (they're worse than dedicated TS models, full stop).
- Calibrated probabilities of numerical events.
- Reasoning about long histories of pure numbers — context length wastes tokens on noise.

## Topics

- Sentiment as a feature: classify headlines/filings, aggregate, lag, join.
- RAG over filings & earnings calls (10-K, 10-Q, 8-K, transcripts).
- Multi-agent orchestration: "analyst" + "skeptic" + "forecaster" roles.
- Structured outputs for forecast rationale.
- Cost & latency: prompt caching, batching, model cascades.

## Files

| File | What it covers |
|------|----------------|
| `01_news_sentiment.py` | Score headlines with Anthropic; aggregate to a daily feature |
| `02_sentiment_as_feature.py` | Add the daily sentiment feature to a 5.3-style LGB |
| `03_earnings_extraction.py` | Pull guidance / numbers from earnings call text via structured outputs |
| `04_rag_filings.py` | Chunk + embed 10-Ks; retrieve passages relevant to a forecast question |
| `05_multi_agent.py` | Analyst + skeptic + forecaster (Anthropic SDK with tool use) |
| `06_forecast_narrative.py` | Turn (point + interval) into a readable explanation |
| `07_cross_provider.py` | Same prompt across Anthropic & OpenAI; compare consistency |
| `08_caching_and_cost.py` | Prompt caching for the recurring "system + history" block |

## Exercises

1. Build a daily SPY headline-sentiment feature for 2020-present and check whether it improves an LGB by walk-forward CV.
2. Use structured outputs to extract `(metric, value, period, source)` tuples from 5 recent 10-Q filings of one ticker.
3. Build a `chunk -> embed -> retrieve` pipeline over the latest 10-K of AAPL; ask "what supply-chain risks are mentioned?"
4. Build a 3-agent loop: analyst proposes a forecast, skeptic argues against, forecaster reconciles. Log the trace.
5. Wire prompt caching for the news-scoring loop and measure cost reduction.

## Resources

- L4.1, L4.2, L4.4 — prerequisites for prompting, RAG, agents.
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- SEC EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=
- `sec-edgar-downloader` for filings: https://pypi.org/project/sec-edgar-downloader/
- BloombergGPT, FinGPT papers — historical context, not used here

## Checkpoint

You can: turn unstructured financial text into a clean numeric feature, validate that it adds signal under walk-forward CV, build an agent that explains a forecast, and reason about LLM cost / latency in a forecasting pipeline.
