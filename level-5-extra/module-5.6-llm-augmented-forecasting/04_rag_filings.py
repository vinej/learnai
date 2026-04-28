"""
04 — RAG over SEC filings.

Workflow:
1. Download a filing (10-K) using sec-edgar-downloader.
2. Chunk into ~800-token segments.
3. Embed with sentence-transformers; index in ChromaDB.
4. Retrieve top-k passages for a question; pass them to Claude as
   context for an answer.

Use cases:
- "What supply-chain risks are mentioned in AAPL's latest 10-K?"
- "How does the company describe FX exposure?"
- "Compare R&D spend trajectory in NVDA's last three 10-Ks."

Run: python 04_rag_filings.py
"""
from __future__ import annotations

from pathlib import Path

from _common import CLAUDE_SMART, anthropic_client

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install: pip install chromadb sentence-transformers sec-edgar-downloader")
    raise SystemExit(0)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def build_index(text: str, name: str = "filing") -> chromadb.Collection:
    chunks = chunk_text(text)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(chunks, batch_size=32, convert_to_numpy=True).tolist()
    client = chromadb.Client()
    coll = client.get_or_create_collection(name=name)
    coll.add(ids=[f"c{i}" for i in range(len(chunks))],
             embeddings=embeddings, documents=chunks)
    return coll


def ask(coll: chromadb.Collection, question: str, k: int = 5) -> str:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = encoder.encode([question]).tolist()
    hits = coll.query(query_embeddings=q_emb, n_results=k)
    context = "\n\n---\n\n".join(hits["documents"][0])
    msg = anthropic_client().messages.create(
        model=CLAUDE_SMART,
        max_tokens=1024,
        system="Answer ONLY from the provided filing excerpts. "
               "Quote short phrases. If the excerpts don't address the question, say so.",
        messages=[{"role": "user", "content":
                   f"Filing excerpts:\n{context}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text


if __name__ == "__main__":
    sample_path = Path(__file__).parent / "sample_filing.txt"
    if sample_path.exists():
        text = sample_path.read_text()
    else:
        # Inline tiny placeholder so the example runs without downloading
        text = """The Company designs, manufactures, and markets smartphones, personal computers,
        tablets, wearables, and accessories, and sells a variety of related services. ... The Company's
        global supply chain is concentrated in China, with secondary nodes in Vietnam and India.
        Geopolitical events in either region could materially affect manufacturing capacity. ...
        Foreign exchange fluctuations are a meaningful component of margin variability. The Company
        hedges a portion of forecast non-USD revenue using forwards and options."""

    coll = build_index(text, name="aapl_10k_excerpt")
    answer = ask(coll, "What supply-chain risks does the filing discuss?")
    print(answer)
