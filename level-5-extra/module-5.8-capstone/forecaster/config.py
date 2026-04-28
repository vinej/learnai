"""Central config — environment, paths, model lists."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CACHE = Path(os.getenv("FORECASTER_CACHE", ROOT / "module-5.8-capstone" / "data_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

MLRUNS = Path(os.getenv("MLFLOW_TRACKING_DIR", ROOT / "module-5.8-capstone" / "mlruns"))


@dataclass(frozen=True)
class Settings:
    anthropic_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    fred_key: str | None = os.getenv("FRED_API_KEY")
    nixtla_key: str | None = os.getenv("NIXTLA_API_KEY")

    claude_fast: str = "claude-haiku-4-5-20251001"
    claude_smart: str = "claude-sonnet-4-6"


SETTINGS = Settings()
