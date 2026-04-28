"""
Exercise 3 — PatchTST on a panel of 8 ETFs.

Train ONE PatchTST on the panel (SPY, QQQ, XLK, XLF, XLE, XLV, XLY,
XLP) — daily log returns, h=5. Compare to:
- per-ticker LightGBM (from 5.3)
- per-ticker LSTM
- panel LightGBM with ticker-id feature

Report a table: rows = tickers, columns = models, cells = MAE.
Average row at the bottom.

Discussion: where does PatchTST help most? Likely the ETFs with
fewer years of data or more idiosyncratic behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
