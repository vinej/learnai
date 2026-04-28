# Setup — Module 5.6

```bash
pip install anthropic openai chromadb sentence-transformers sec-edgar-downloader feedparser
```

API keys in `level-5-extra/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The shared loaders from Module 5.1 are reused.

## A note on cost

These examples make real API calls. Most run for under $0.50 each
using cheap models (Claude Haiku, GPT-4o-mini). Where heavier models
are needed, the file calls them out explicitly.

Always review the model and quantity of calls in a script before
running it on a large date range.
