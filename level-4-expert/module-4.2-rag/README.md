# Module 4.2 — Retrieval-Augmented Generation (RAG)

**Level:** 4 — Expert
**Estimated time:** 2-3 weeks

## Goal
Build LLM apps that answer questions over private/external data accurately and verifiably.

## Topics
### Embeddings
- What an embedding is, geometrically
- Embedding models: OpenAI, Voyage, Cohere, sentence-transformers
- Choosing dimension, normalization, cosine vs dot product

### Vector stores
- **FAISS**, **Chroma**, **pgvector**, **Pinecone**, **Weaviate**, **Qdrant**, **Milvus**
- Tradeoffs: managed vs self-hosted, ANN index types (HNSW, IVF)
- Metadata filtering, hybrid search (BM25 + vector)

### Document pipeline
- Loaders: PDFs, HTML, Markdown, Office docs
- Chunking strategies: fixed size, semantic, recursive, parent-document
- Cleaning: deduplication, table extraction, OCR

### Retrieval
- Top-k similarity search
- Re-ranking with cross-encoders (Cohere Rerank, bge-reranker)
- Query rewriting (HyDE, multi-query)
- Hybrid search combining lexical + semantic

### Generation
- Prompt templates that cite sources
- Handling "I don't know"
- Streaming with retrieved context

### Evaluation
- Faithfulness, answer relevance, context precision/recall
- **Ragas**, **TruLens**, **DeepEval**
- LLM-as-judge with rubrics
- Building golden eval sets

### Frameworks (and when to skip them)
- **LangChain**, **LlamaIndex**, **Haystack**
- When a 50-line script beats them all

## Exercises
1. Build a simple RAG over your own Markdown notes using FAISS + the Anthropic SDK.
2. Add a reranker; measure precision@3 improvement on a labeled eval set.
3. Compare 3 chunking strategies on retrieval quality.
4. Build a Ragas eval suite with 50+ questions and run it on every code change.
5. Migrate from FAISS to pgvector; benchmark latency.

## Resources
- LlamaIndex docs
- Pinecone Learning Center
- "RAG From Scratch" by LangChain (YouTube)
- Ragas docs: https://docs.ragas.io/

## Checkpoint
You can ship a RAG app over a real corpus, with measurable retrieval quality, re-ranking, citations, and an evaluation suite that runs in CI.
