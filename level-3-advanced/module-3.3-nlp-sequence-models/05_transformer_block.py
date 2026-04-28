"""
05 — A transformer block, end to end

A "block" is the unit you stack to build a transformer. Pseudocode:

    x' = x + MultiHeadAttention(LayerNorm(x))     # attention sublayer
    y  = x' + FFN(LayerNorm(x'))                  # feed-forward sublayer

The two key tricks:
  RESIDUAL CONNECTIONS (the `x +`) — keep gradients flowing through depth.
  LAYER NORM — normalize per-sample, not per-batch. Stabilizes training.
  FFN — usually two Linears with GELU/SiLU between them. Larger inner dim
        than d_model (typically 4× — that's where most params live).

Two style variants:
  POST-NORM (original "Attention is all you need")
    x' = LayerNorm(x + Attention(x))
  PRE-NORM (modern: GPT-2/3, Llama, etc.)
    x' = x + Attention(LayerNorm(x))     ← we use this
  Pre-norm trains more reliably at depth.

Run: python 05_transformer_block.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Multi-head attention (same as file 04)
# ───────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        # F.scaled_dot_product_attention is the fused fast path (PyTorch ≥ 2)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


# ───────────────────────────────────────────────────────────
# Feed-forward: two-layer MLP applied per token
# ───────────────────────────────────────────────────────────
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ───────────────────────────────────────────────────────────
# A complete encoder block (pre-norm)
# ───────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop2(self.ffn (self.norm2(x)))
        return x


# ───────────────────────────────────────────────────────────
# Putting it together: a tiny encoder
# ───────────────────────────────────────────────────────────
class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids, mask=None):
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(positions)        # token + position
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        return self.head(x)


vocab_size = 1000
model = TinyEncoder(vocab_size=vocab_size, max_len=64)
ids = torch.randint(0, vocab_size, (2, 16))
logits = model(ids)
print(f"input ids: {tuple(ids.shape)}")
print(f"output logits: {tuple(logits.shape)}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Encoder vs decoder vs encoder-decoder
# ───────────────────────────────────────────────────────────
# ENCODER (BERT-family): full bidirectional attention. Best for understanding
#   tasks (classification, NER, similarity).
# DECODER (GPT-family): causal mask. Best for generation (LMs, chat).
# ENCODER-DECODER (T5, BART, mT5): encoder reads input, decoder generates
#   output, with cross-attention between them. Best for translation,
#   summarization, structured generation.
#
# Architecturally they share these blocks. The differences are:
#   • Causal mask in decoders.
#   • Cross-attention layers in the decoder of enc-dec models.
#   • Different training objectives (MLM vs causal LM vs span corruption).


# ───────────────────────────────────────────────────────────
# Positional encoding
# ───────────────────────────────────────────────────────────
# Attention is order-AGNOSTIC. We must inject position somehow:
#   • LEARNED — nn.Embedding(seq_len, d). What we used above. Simple.
#   • SINUSOIDAL — fixed sin/cos. The original transformer used this.
#   • ROTARY (RoPE) — rotates Q/K. Llama, GPT-NeoX. Generalizes to longer seqs.
#   • ALiBi — adds a position-distance bias to attention scores.
# All work; modern LLMs prefer RoPE for length generalization.
