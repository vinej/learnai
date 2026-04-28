"""
Exercise 2 — Build a tiny transformer block from scratch and train it

You'll build the block in PyTorch (multi-head attention + FFN + residual +
layer norm) and train a 2-layer encoder-style model on a toy task: predict
the LAST character of a fixed-length string given the first n-1 characters,
where the last character is determined by simple combinatorial rules.

Specifically: predict the SUM (mod 10) of the digits in the input.

The script asserts validation accuracy ≥ 95%.

Run: python exercises/02_tiny_transformer.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Synthetic task: digit-sum mod 10
# ───────────────────────────────────────────────────────────
SEQ_LEN = 8
VOCAB   = 10           # digits 0..9


def make_batch(batch_size: int):
    """Return (X, y) where y = sum(X) mod 10."""
    X = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)
    y = X.sum(dim=1) % VOCAB
    return X, y


# ───────────────────────────────────────────────────────────
# Build the transformer block (yours to verify against)
# ───────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # split heads
        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_k).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int,
                 d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks  = nn.Sequential(*[TransformerBlock(d_model, n_heads)
                                       for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.norm(x)
        # Pool by mean over sequence, then classify.
        return self.head(x.mean(dim=1))


model = TinyTransformer(vocab_size=VOCAB, max_len=SEQ_LEN).to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
criterion = nn.CrossEntropyLoss()

STEPS = 2000
for step in range(STEPS):
    x, y = make_batch(128)
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    if step % 250 == 0:
        with torch.no_grad():
            xv, yv = make_batch(2048)
            acc = (model(xv).argmax(dim=1) == yv).float().mean().item()
        print(f"step {step:>4}: loss={loss.item():.3f}  val_acc={acc:.3f}")


# ───────────────────────────────────────────────────────────
# Final eval
# ───────────────────────────────────────────────────────────
with torch.no_grad():
    xv, yv = make_batch(8192)
    final_acc = (model(xv).argmax(dim=1) == yv).float().mean().item()

print(f"\nfinal val accuracy: {final_acc:.4f}")
assert final_acc >= 0.95, f"target ≥ 0.95, got {final_acc:.4f}"
print("tiny transformer learns digit-sum-mod-10 ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace the synthetic task with sequence reversal: predict X[::-1].
#   Now the model needs to attend to specific positions, not just the sum.
# • Add a CAUSAL mask and turn the model into an autoregressive char LM.
# • Compare convergence vs an RNN of similar param count.
# ─────────────────────────────────────────────────────────────────────────────
