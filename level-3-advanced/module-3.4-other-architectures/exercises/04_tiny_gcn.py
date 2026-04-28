"""
Exercise 4 — Train a tiny GCN on a synthetic clustered graph

Build a 3-community stochastic block model graph, label only 5 nodes per
class, train a 2-layer GCN to classify all the others.

The script asserts test accuracy ≥ 0.85 on the unlabeled nodes.

Run: python exercises/04_tiny_gcn.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
np.random.seed(0)


# ───────────────────────────────────────────────────────────
# Graph
# ───────────────────────────────────────────────────────────
sizes = [25, 25, 25]               # 3 communities of 25 nodes
N = sum(sizes)
p_in, p_out = 0.25, 0.02
G = nx.stochastic_block_model(sizes,
                              [[p_in, p_out, p_out],
                               [p_out, p_in, p_out],
                               [p_out, p_out, p_in]],
                              seed=0)
y = np.array([G.nodes[n]["block"] for n in G.nodes])
print(f"graph: {N} nodes, {G.number_of_edges()} edges")


# Symmetrically-normalized adjacency with self-loops
A = nx.to_numpy_array(G).astype(np.float32)
A_hat = A + np.eye(N)
deg = A_hat.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
A_norm = torch.from_numpy(D_inv_sqrt @ A_hat @ D_inv_sqrt)


# ───────────────────────────────────────────────────────────
# Features: identity matrix (no real features given)
# ───────────────────────────────────────────────────────────
H0 = torch.eye(N)


# ───────────────────────────────────────────────────────────
# 2-layer GCN
# ───────────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout=0.5):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=False)
        self.W2 = nn.Linear(hidden, n_classes, bias=False)
        self.dropout = dropout

    def forward(self, x, A):
        h = F.relu(A @ self.W1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return A @ self.W2(h)


model = GCN(in_dim=N, hidden=16, n_classes=3)


# ───────────────────────────────────────────────────────────
# Semi-supervised setup: 5 labels per class
# ───────────────────────────────────────────────────────────
labels = torch.from_numpy(y)
train_mask = torch.zeros(N, dtype=torch.bool)
for c in range(3):
    cls_idx = np.where(y == c)[0][:5]
    train_mask[cls_idx] = True
test_mask = ~train_mask
print(f"labeled: {train_mask.sum().item()} / {N}, "
      f"test: {test_mask.sum().item()}")


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-4)
for epoch in range(300):
    model.train()
    opt.zero_grad()
    logits = model(H0, A_norm)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    opt.step()

    if epoch % 50 == 0 or epoch == 299:
        model.eval()
        with torch.no_grad():
            preds = model(H0, A_norm).argmax(dim=1)
            tr_acc = (preds[train_mask] == labels[train_mask]).float().mean().item()
            te_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
        print(f"epoch {epoch:>3}  loss={loss.item():.3f}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")


# ───────────────────────────────────────────────────────────
# Final eval + visualization
# ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    final_preds = model(H0, A_norm).argmax(dim=1).numpy()
final_acc = (final_preds[test_mask.numpy()] == y[test_mask.numpy()]).mean()
print(f"\nfinal test accuracy: {final_acc:.4f}")


# Visualize
pos = nx.spring_layout(G, seed=0)
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
nx.draw(G, pos, ax=axes[0], node_color=y, cmap="tab10", node_size=70,
        edge_color="lightgrey")
axes[0].set_title("True classes")

nx.draw(G, pos, ax=axes[1], node_color=final_preds, cmap="tab10", node_size=70,
        edge_color="lightgrey")
axes[1].set_title("GCN predictions")

# Highlight the labeled training nodes
node_colors = ["red" if train_mask[i] else "lightblue" for i in range(N)]
nx.draw(G, pos, ax=axes[2], node_color=node_colors, node_size=70,
        edge_color="lightgrey")
axes[2].set_title("Labeled (red) vs unlabeled (blue)")

fig.tight_layout()
out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "exercise_04_gcn.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / 'exercise_04_gcn.png'}")


assert final_acc >= 0.85, f"target ≥ 0.85, got {final_acc:.4f}"
print("tiny GCN passes ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Reduce labels to 2 per class. Does the GCN still work? (It often does —
#   that's the strength of message-passing on connected graphs.)
# • Replace the GCN layer with a GAT (graph attention) layer — implement
#   attention-weighted neighbour aggregation.
# • Try the real Cora / Citeseer dataset via `torch_geometric` (heavier install).
# ─────────────────────────────────────────────────────────────────────────────
