# Setup — Module 4.1 (LLMs & Prompt Engineering)

This module uses the **Anthropic Python SDK** to call Claude. Some scripts run fully offline (anatomy, sampling demos, cross-provider concepts); the API-using ones require a key.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `anthropic`    | Anthropic Python SDK (Claude)                    |
| `pydantic`     | Validate structured LLM outputs                  |
| `Pillow`       | Encode images for vision requests                |
| `numpy`, `matplotlib` | Math + plots                              |

## 4. Get an Anthropic API key

1. Go to https://console.anthropic.com → **API Keys** → **Create Key**.
2. Copy the key.
3. Set it in your shell:

   ```bash
   # Windows PowerShell (current session)
   $env:ANTHROPIC_API_KEY = "sk-ant-..."

   # bash/zsh
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

Or add it to a `.env` file at the project root and load it via `python-dotenv` or your shell.

## 5. Cost expectations

Every API call costs money. The scripts default to **Claude Haiku 4.5** — the cheapest production-grade model. Running every script in this module end-to-end should cost a few cents.

Want to upgrade?
- `claude-sonnet-4-6` — better reasoning, ~5× the cost.
- `claude-opus-4-7` — top tier, ~25× the cost. Use for difficult tasks only.

## 6. Run the lessons

```bash
python 01_llm_anatomy.py                 # offline
python 02_sampling_and_cost.py           # offline
python 03_anthropic_basics.py            # API
python 04_prompt_patterns.py             # API
python 05_structured_outputs.py          # API
python 06_prompt_caching.py              # API
python 07_tool_use.py                    # API
python 08_vision_and_pdfs.py             # API
python 09_cross_provider.py              # offline
```

## 7. Run the exercises

```bash
python exercises/01_cli_chatbot.py
python exercises/02_prompt_caching_verify.py
python exercises/03_structured_extraction.py
python exercises/04_temperature_diversity.py
python exercises/05_few_shot_classifier.py
```

If `ANTHROPIC_API_KEY` is missing, API-using scripts print a friendly error and exit cleanly.

## Tips

- **Cache long system prompts** when you'll reuse them. The cache_control on content blocks gives you up to 90% discount on cached tokens.
- **Default to streaming** for any user-facing UI. The first token arrives in ~300ms vs waiting for the full response.
- **Track tokens & dollars** from day one. `response.usage` has everything; log it to MLflow / a CSV / Langfuse.
- **Test with the cheapest model** that does the job. Don't reach for Opus until Haiku has obviously failed.
