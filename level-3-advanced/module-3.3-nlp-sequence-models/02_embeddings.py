"""
02 — Word and token embeddings

After tokenization, you have integers. Models can't compute much with raw
ids, so we map each id to a learned VECTOR.

  one-hot encoding  — vocab×1 sparse, no notion of similarity.
  random embedding  — vocab×D learned during training.
  pretrained        — word2vec / GloVe (static) → BERT (contextual).

In modern transformers, the embedding layer is the FIRST lookup: id → vector,
and these vectors are LEARNED PARAMETERS — same as everything else.

Run: python 02_embeddings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# nn.Embedding: a (vocab × dim) lookup table
# ───────────────────────────────────────────────────────────
vocab_size = 10
embed_dim  = 4
embedding = nn.Embedding(vocab_size, embed_dim)
print("embedding weight shape:", embedding.weight.shape)

ids = torch.tensor([2, 7, 1, 7])
print("\nembed(ids):")
print(embedding(ids))
# It's just `embedding.weight[ids]` under the hood.


# ───────────────────────────────────────────────────────────
# Train embeddings to capture similarity
# ───────────────────────────────────────────────────────────
# Toy task: predict the next token in a tiny corpus. Embeddings learn that
# context-similar tokens should be near each other.
sentences = [
    ["the", "cat", "sat", "on", "mat"],
    ["a",   "cat", "ran", "on", "rug"],
    ["the", "dog", "sat", "on", "mat"],
    ["a",   "dog", "ran", "on", "rug"],
    ["the", "cat", "and", "the", "dog"],
]

# Build a vocab
vocab = sorted({w for s in sentences for w in s})
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}
V = len(vocab)
print(f"\nvocab: {vocab}")


# Tiny model: predict the *center* word from its neighbours (CBOW-style).
# Each training pair = ((w_left, w_right), w_center).
def make_pairs(sents):
    Xs, ys = [], []
    for s in sents:
        ids = [word_to_id[w] for w in s]
        for i in range(1, len(ids) - 1):
            Xs.append([ids[i - 1], ids[i + 1]])
            ys.append(ids[i])
    return torch.tensor(Xs), torch.tensor(ys)


X, y = make_pairs(sentences)
print(f"training pairs: {len(X)}")


class Cbow(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, ctx_ids):
        # ctx_ids: (B, 2). Average the neighbours.
        emb = self.embed(ctx_ids).mean(dim=1)
        return self.linear(emb)


model = Cbow(V, embed_dim=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(2000):
    opt.zero_grad()
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
print(f"final CBOW loss: {loss.item():.4f}")


# ───────────────────────────────────────────────────────────
# Inspect the learned vectors
# ───────────────────────────────────────────────────────────
embeddings = model.embed.weight.detach().numpy()


def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# Words that play similar roles in our corpus should be near each other.
print(f"\ncos('cat', 'dog')  = {cosine_sim(embeddings[word_to_id['cat']], embeddings[word_to_id['dog']]):+.3f}")
print(f"cos('cat', 'mat')  = {cosine_sim(embeddings[word_to_id['cat']], embeddings[word_to_id['mat']]):+.3f}")
print(f"cos('mat', 'rug')  = {cosine_sim(embeddings[word_to_id['mat']], embeddings[word_to_id['rug']]):+.3f}")
print(f"cos('the', 'a')    = {cosine_sim(embeddings[word_to_id['the']], embeddings[word_to_id['a']]):+.3f}")


# ───────────────────────────────────────────────────────────
# Visualize via PCA
# ───────────────────────────────────────────────────────────
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pca[:, 0], pca[:, 1], s=30)
for i, w in enumerate(vocab):
    ax.annotate(w, (pca[i, 0], pca[i, 1]),
                xytext=(5, 5), textcoords="offset points")
ax.set_title("Learned 8-D word embeddings (PCA → 2D)")
ax.grid(True, alpha=0.3)
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "02_embeddings.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '02_embeddings.png'}")


# ───────────────────────────────────────────────────────────
# A short history of the field
# ───────────────────────────────────────────────────────────
# • word2vec (2013)  — train embeddings on giant corpora; "king − man + woman ≈ queen".
# • GloVe (2014)      — global co-occurrence statistics. Very strong static vectors.
# • fastText (2016)   — subword n-grams; handles OOV.
# • ELMo (2018)       — contextual embeddings: "bank" near a river ≠ "bank" with money.
# • BERT (2018)       — bidirectional contextual embeddings via transformers.
# • Modern LLMs       — token embeddings are still the first layer.
#
# In practice today, you almost never train static word vectors anymore.
# You use the SUBWORD TOKEN EMBEDDINGS that come with a pretrained model.
