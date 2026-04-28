# Setup — Module 3.3 (NLP & Sequence Models)

This module covers tokenization, RNNs/LSTMs, the transformer architecture, and Hugging Face's `transformers` + `datasets` ecosystem.

> **Disk warning:** the pretrained models downloaded by Hugging Face land in `~/.cache/huggingface/`. Plan for ~1-2 GB total across DistilBERT, T5-small, and a sentence-transformer.

## 1. Python ≥ 3.11 + PyTorch

If you completed Module 3.1, you're set. Otherwise see [../module-3.1-deep-learning-foundations/INSTALL.md](../module-3.1-deep-learning-foundations/INSTALL.md).

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

| Package                | Purpose                                          |
|------------------------|--------------------------------------------------|
| `torch`                | The framework                                    |
| `transformers`         | Hugging Face models, tokenizers, Trainer         |
| `datasets`             | Hugging Face datasets library                    |
| `tokenizers`           | Fast tokenizers (BPE, WordPiece, SentencePiece) |
| `sentence-transformers`| Sentence embeddings for retrieval                |
| `evaluate`             | Standard NLP metrics (ROUGE, accuracy, F1)       |
| `rouge_score`          | Backend for ROUGE                                |
| `numpy`, `matplotlib`  | Math + plots                                     |

## 4. Run the lessons

```bash
python 01_tokenization.py
python 02_embeddings.py
python 03_rnn_lstm_gru.py
python 04_attention_mechanism.py
python 05_transformer_block.py
python 06_huggingface_basics.py        # downloads ~250MB DistilBERT on first run
python 07_finetune_classification.py   # uses the cached DistilBERT
python 08_sentence_embeddings.py       # downloads ~80MB MiniLM on first run
```

Plotting scripts save PNGs to `figures/`.

## 5. Run the exercises

```bash
python exercises/01_char_rnn_shakespeare.py
python exercises/02_tiny_transformer.py
python exercises/03_finetune_imdb.py             # ~5-10 min on CPU
python exercises/04_finetune_t5_summarization.py # ~10-20 min on CPU
python exercises/05_semantic_search.py
```

## CPU vs GPU

Everything in this module runs on CPU. With a GPU, fine-tuning exercises drop from minutes to seconds. The exercises use *small subsets* and *small models* so CPU users aren't punished.

## Setting Hugging Face cache location

If your home drive is small, point the cache elsewhere:

```bash
# Windows PowerShell
$env:HF_HOME = "D:\hf_cache"
# bash
export HF_HOME="/data/hf_cache"
```

## Tip

You'll write `from transformers import AutoTokenizer, AutoModelForSequenceClassification` more times than you'll write Python `import` statements over the rest of your career. Get fluent with the `Auto*` family — they pick the right tokenizer/model class for any model on the Hub.
