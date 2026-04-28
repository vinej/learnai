# Setup — Module 4.2 (Retrieval-Augmented Generation)

RAG = give an LLM relevant context retrieved from your data so it answers grounded questions instead of hallucinating.

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

| Package                 | Purpose                                          |
|-------------------------|--------------------------------------------------|
| `anthropic`             | Claude for the generation step                   |
| `sentence-transformers` | Local embedding + cross-encoder reranker         |
| `faiss-cpu`             | Vector index                                     |
| `rank-bm25`             | Lexical search for hybrid retrieval              |
| `pydantic`              | Validation                                       |
| `numpy`, `matplotlib`   | Math + plots                                     |

> **First-run downloads:** sentence-transformers will fetch ~80MB of weights (`all-MiniLM-L6-v2`) and ~120MB for the cross-encoder.

## 4. ANTHROPIC_API_KEY

Required for any script that generates an answer. See [Module 4.1's INSTALL.md](../module-4.1-llms-prompt-engineering/INSTALL.md) for setup.

Scripts that JUST do retrieval (embeddings, FAISS, chunking) work without an API key.

## 5. Run the lessons

```bash
python 01_embeddings.py            # offline
python 02_chunking.py              # offline
python 03_faiss.py                 # offline
python 04_rag_pipeline.py          # API
python 05_hybrid_and_reranking.py  # offline
python 06_query_rewriting.py       # API
python 07_evaluation.py            # API (LLM-as-judge)
python 08_frameworks.py            # offline
```

## 6. Run the exercises

```bash
python exercises/01_notes_rag.py            # API
python exercises/02_reranker_lift.py        # offline
python exercises/03_chunking_comparison.py  # offline
python exercises/04_golden_set_eval.py      # API
python exercises/05_faiss_indexes.py        # offline; benchmarks Flat/HNSW/IVF
```

## Tip

The biggest gains in RAG are upstream of the model: **better chunks, better retrieval, better reranking**. A great retrieval stack with a Haiku-class generator beats sloppy retrieval with Opus.
