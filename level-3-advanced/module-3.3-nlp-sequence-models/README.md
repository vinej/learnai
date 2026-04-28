# Module 3.3 — NLP & Sequence Models

**Level:** 3 — Advanced
**Estimated time:** 3 weeks

## Goal
Understand how text is represented and processed by neural networks, from word vectors to transformers.

## Topics
### Text representation
- Tokenization: word, subword (BPE, WordPiece, SentencePiece)
- Vocab, special tokens (`[CLS]`, `[SEP]`, `<|endoftext|>`)
- Embeddings: word2vec, GloVe, FastText
- Contextual embeddings (BERT-family)

### Sequence models
- RNN, LSTM, GRU — and why they fall short on long sequences
- Sequence-to-sequence with attention

### Transformers (the heart of modern NLP)
- Self-attention: queries, keys, values
- Multi-head attention
- Positional encodings (sinusoidal, learned, RoPE, ALiBi)
- Encoder vs decoder vs encoder-decoder
- The architectures: BERT (encoder), GPT (decoder), T5 (enc-dec)

### Hugging Face stack
- `transformers`: `AutoModel`, `AutoTokenizer`, pipelines
- `datasets` library
- Fine-tuning with `Trainer` and with raw PyTorch loops
- Pushing models to the Hub

### Tasks
- Text classification (sentiment, topic)
- Named-entity recognition (NER)
- Question answering
- Summarization
- Translation

## Exercises
1. Implement a character-level RNN that generates Shakespeare-like text.
2. Implement a tiny transformer block (attention + FFN + residual) from scratch.
3. Fine-tune `bert-base-uncased` for sentiment classification on IMDb.
4. Fine-tune T5 for a summarization task; measure ROUGE.
5. Use sentence-transformers to build a semantic search over your own documents.

## Resources
- "The Illustrated Transformer" — Jay Alammar
- "The Annotated Transformer" — Harvard NLP
- Hugging Face NLP course (free): https://huggingface.co/learn/nlp-course
- Andrej Karpathy's "Let's build GPT" (YouTube)

## Checkpoint
You can explain self-attention to a colleague using only matrix shapes, and you can fine-tune a transformer on a custom text dataset.
