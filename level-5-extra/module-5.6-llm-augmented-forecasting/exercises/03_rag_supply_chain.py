"""
Exercise 3 — RAG over AAPL's latest 10-K.

1. Download the latest AAPL 10-K (sec-edgar-downloader).
2. Chunk + embed with sentence-transformers.
3. Index in Chroma.
4. Run these queries:
   - "What supply-chain risks are mentioned?"
   - "How does the company describe foreign exchange hedging?"
   - "What is the company's R&D spend trajectory?"
   - "What share-repurchase activity has been authorized?"

Show the retrieved excerpts and Claude's grounded answer.
Discuss two failure modes you expect: missing-context and over-quoting.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement.
