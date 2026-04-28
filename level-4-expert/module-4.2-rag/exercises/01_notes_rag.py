"""
Exercise 1 — RAG over your own Markdown notes

Point this script at a folder of `.md` files. It will:
  1. Load every Markdown file.
  2. Recursively chunk each file (paragraph → sentence → fixed).
  3. Embed all chunks with sentence-transformers.
  4. Build a FAISS index.
  5. Answer questions about the notes with citations.

Default: uses the bundled `sample_notes/` folder.
You can pass `--notes <path>` to point at your own.

Run: python exercises/01_notes_rag.py
     python exercises/01_notes_rag.py --notes ~/my-notes
"""

import argparse
import re
import sys
from pathlib import Path

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, print_usage, require_api_key


HERE = Path(__file__).parent
SAMPLE_NOTES = HERE / "sample_notes"


def ensure_sample_notes():
    """Create a tiny notes folder if it doesn't exist yet."""
    if SAMPLE_NOTES.exists():
        return
    SAMPLE_NOTES.mkdir(parents=True)
    (SAMPLE_NOTES / "python.md").write_text("""\
# Python language notes

Python is dynamically typed. Type hints (PEP 484) are optional but
help with tooling.

## Comprehensions
List comprehensions are the idiomatic way to transform a list.
Generator expressions are similar but lazy.

## Async
asyncio is built-in. Coroutines defined with `async def` are run via
`asyncio.run` or a running loop. Use `await` only inside an async function.
""")
    (SAMPLE_NOTES / "ml.md").write_text("""\
# Machine learning recap

## Classification metrics
Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 is their harmonic
mean. ROC-AUC measures threshold-free ranking quality.

## Pipelines
sklearn Pipelines glue preprocessing and a model into one estimator. They
are leakage-safe inside cross-validation.

## Cross-validation
For time series, use TimeSeriesSplit so the train slice is always before
the test slice. Random shuffling leaks the future.
""")
    (SAMPLE_NOTES / "git.md").write_text("""\
# Git cheat sheet

## Everyday
- git status: see changes
- git add FILE: stage a change
- git commit -m "msg": commit staged
- git diff [--staged]: inspect

## Branches
git switch -c feature/x creates a new branch. Push with `git push -u origin feature/x`.

## Undo
- git restore FILE: discard local changes
- git revert SHA: create a commit that undoes SHA (safe, public)
- git reset --hard HEAD~1: drop the last commit AND its changes (destructive)
""")
    print(f"created sample notes in: {SAMPLE_NOTES}")


# ───────────────────────────────────────────────────────────
# Recursive chunker — paragraph → sentence → fixed
# ───────────────────────────────────────────────────────────
def chunk(text: str, max_size: int = 400) -> list[str]:
    out, paragraphs = [], [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in paragraphs:
        if len(p) <= max_size:
            out.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) > max_size:
                if buf:
                    out.append(buf.strip())
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf:
            out.append(buf.strip())
    return out


def load_notes(notes_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(notes_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, c in enumerate(chunk(text)):
            chunks.append({
                "id": f"{path.stem}#{i}",
                "source": str(path.relative_to(notes_dir)),
                "text": c,
            })
    return chunks


# ───────────────────────────────────────────────────────────
# Build index + ask
# ───────────────────────────────────────────────────────────
def build_index(chunks: list[dict]):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = embedder.encode([c["text"] for c in chunks],
                          normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return embedder, index


SYSTEM_PROMPT = """\
You answer questions using ONLY the provided notes. Cite chunk IDs in
[brackets]. If the notes don't contain the answer, say
"I don't know based on the provided notes."
Be brief.
"""


def answer(client, embedder, index, chunks, query: str, k: int = 4):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, idx = index.search(q_emb, k)
    hits = [chunks[i] for i in idx[0]]
    docs_xml = "\n".join(f"<doc id=\"{h['id']}\" source=\"{h['source']}\">{h['text']}</doc>"
                         for h in hits)
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"<documents>\n{docs_xml}\n</documents>\n\nQuestion: {query}"}],
    )
    print_usage(query[:30], response.usage)
    return response.content[0].text.strip(), hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default=str(SAMPLE_NOTES),
                        help="path to a folder of Markdown files")
    args = parser.parse_args()

    ensure_sample_notes()
    require_api_key()

    notes_dir = Path(args.notes).expanduser()
    chunks = load_notes(notes_dir)
    print(f"loaded {len(chunks)} chunks from {notes_dir}")

    embedder, index = build_index(chunks)
    client = anthropic.Anthropic()

    questions = [
        "How do I undo a commit safely?",
        "What's the difference between precision and recall?",
        "How do I run an async function in Python?",
        "What's a transformer?",        # not in notes — should say IDK
    ]
    for q in questions:
        print(f"\nQ: {q}")
        ans, hits = answer(client, embedder, index, chunks, q)
        print("retrieved:")
        for h in hits:
            print(f"  - {h['id']}  ({h['source']})")
        print("answer:", ans)


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add an interactive REPL: `while True: input("ask> ") → answer`.
# • Persist the FAISS index to disk and skip rebuilding when notes haven't changed.
# • Add a `--stream` flag and stream the answer.
# • Switch to parent-document retrieval: embed small chunks, return the parent
#   ## section as context.
# ─────────────────────────────────────────────────────────────────────────────
