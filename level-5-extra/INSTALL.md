# Setup — Level 5

This level uses many third-party packages. Each sub-module ships its own `requirements.txt` so you can install only what you need, but you can also create one shared environment for the whole level.

## 1. Python version

Python **3.11 or 3.12**. Some deep-learning and foundation-model packages lag on 3.13 as of early 2026.

## 2. Shared virtual environment (recommended)

```bash
# from the level-5-extra/ folder
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer per-module envs, `cd` into each sub-module and follow its own `INSTALL.md`.

## 3. API keys

Create a `.env` file at `level-5-extra/.env` (already in `.gitignore` — never commit keys):

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
```

- **Anthropic** — get one at https://console.anthropic.com/
- **OpenAI** — get one at https://platform.openai.com/api-keys
- **FRED** — free, https://fred.stlouisfed.org/docs/api/api_key.html

Examples load these via `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

## 4. GPU (optional but useful)

Modules 5.4 and 5.5 train transformers; CPU works for examples but is slow.

```bash
# Check
python -c "import torch; print(torch.cuda.is_available())"
```

If you don't have a GPU, you can:
- Use the foundation-model APIs in 5.5 (no local GPU needed).
- Reduce sequence lengths and epochs in 5.4 examples.
- Use Colab / Modal / Lightning Studios for the heavier exercises.

## 5. Verify install

```bash
python verify_install.py
```

This script checks all major packages, both API keys, and basic data downloads from `yfinance` and FRED.
