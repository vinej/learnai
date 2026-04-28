"""
04 — Self-attention, by hand

The single most important idea in modern NLP:

  attention(Q, K, V) = softmax(Q K^T / √d_k) V

Each token in the sequence asks "who else here matters to me?" by computing
similarity (Q · K^T) with every other token, then takes a weighted average
of their values (V). Every token attends to every other token in parallel —
no recurrence, fully GPU-friendly.

Q, K, V are all derived from the same input X via three linear projections
(W_Q, W_K, W_V). That's "self-" attention.

Run: python 04_attention_mechanism.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Scaled dot-product attention, from scratch
# ───────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (B, T, d). Returns (B, T, d)."""
    d_k = Q.size(-1)
    # similarity scores
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)         # (B, T_q, T_k)

    if mask is not None:
        # mask: 1 where allowed, 0 where forbidden
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)                       # (B, T_q, T_k)
    out = weights @ V                                         # (B, T_q, d)
    return out, weights


# ───────────────────────────────────────────────────────────
# Build Q, K, V from a sequence X
# ───────────────────────────────────────────────────────────
B, T, D = 1, 5, 8                       # batch, tokens, model dim
X = torch.randn(B, T, D)

W_Q = nn.Linear(D, D, bias=False)
W_K = nn.Linear(D, D, bias=False)
W_V = nn.Linear(D, D, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)
print(f"Q, K, V shapes: {tuple(Q.shape)}")

out, weights = scaled_dot_product_attention(Q, K, V)
print(f"out shape: {tuple(out.shape)}")
print(f"attention weights for sample 0:")
print(weights[0].round(decimals=3))
# Each row sums to 1 (softmax). Each row is "how much token i attends to each j".


# ───────────────────────────────────────────────────────────
# Causal masking — for decoder / generation
# ───────────────────────────────────────────────────────────
# In language modeling, token at position t must NOT see future tokens t+1, t+2...
# A lower-triangular mask enforces that.
causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0)         # (1, T, T)
print(f"\ncausal mask:")
print(causal_mask.squeeze().int())

masked_out, masked_w = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\nmasked attention weights (note upper-triangular zeros):")
print(masked_w[0].round(decimals=3))


# ───────────────────────────────────────────────────────────
# Multi-head attention
# ───────────────────────────────────────────────────────────
# Split D into h smaller heads, run attention independently in each, then
# concatenate. Each head can specialize in a different relationship.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads

        # One projection each — split into h heads inside forward.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        # Project + reshape to (B, h, T, d_k)
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                       # (B, h, T, d_k)

        # Concatenate heads → (B, T, D), then project
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


mha = MultiHeadAttention(d_model=64, n_heads=8)
y = mha(torch.randn(2, 10, 64))
print(f"\nMHA output shape: {tuple(y.shape)}")


# ───────────────────────────────────────────────────────────
# PyTorch's built-in
# ───────────────────────────────────────────────────────────
# nn.MultiheadAttention exists; it's flexible but the API can be confusing.
# F.scaled_dot_product_attention (PyTorch 2.0+) is the modern standard —
# fused implementation that uses Flash Attention on supported hardware.
out = F.scaled_dot_product_attention(Q, K, V)
print(f"\nF.scaled_dot_product_attention output: {tuple(out.shape)}")


# ───────────────────────────────────────────────────────────
# Why it matters
# ───────────────────────────────────────────────────────────
# • Cost is O(T²) — quadratic in sequence length. That's why long-context
#   LLMs are hard. Modern tricks: Flash Attention (faster, exact), sparse
#   attention, sliding-window attention (Mistral), state-space models (Mamba).
# • Each head can attend to different things: syntax, position, semantic.
# • Stacking attention + FFN blocks gives you the transformer (next file).
