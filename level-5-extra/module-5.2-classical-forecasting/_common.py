"""
Shared imports for module 5.2 — re-exports the data loaders from 5.1
so this folder doesn't need its own cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

_M51 = Path(__file__).resolve().parents[1] / "module-5.1-financial-timeseries-foundations"
if str(_M51) not in sys.path:
    sys.path.insert(0, str(_M51))

from _common import fetch_market, fetch_fred  # noqa: E402,F401  (re-export)
