"""
06 — Graph Neural Networks (GNNs)

A GRAPH has nodes (e.g. users, atoms, papers) and edges (relationships).
GNNs learn node representations by REPEATEDLY AGGREGATING features from
each node's neighbours — "message passing".

Simplest variant — the GCN layer (Kipf & Welling 2017):

  H' = σ(  Â · H · W  )

where Â is a normalized adjacency matrix and H is the node-feature matrix.
Each node's new feature is a learned mixture of itself and its neighbours.

Use cases:
  • Fraud detection (a node = a transaction; edges = shared device/IP)
  • Drug discovery (molecule = graph of atoms)
  • Social networks, recommenders, citation networks, code analysis.

This file builds a 2-layer GCN from scratch and trains it on a synthetic
clustered graph.

Run: python 06_graph_neural_networks.py
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
# Synthetic graph: stochastic block model with 3 communities
# ───────────────────────────────────────────────────────────
# 60 nodes, 3 classes. Edges are dense WITHIN class, sparse between classes.
sizes = [20, 20, 20]
p_in  = 0.3        # prob of edge inside a community
p_out = 0.02       # prob of edge across communities

G = nx.stochastic_block_model(sizes, [[p_in, p_out, p_out],
                                       [p_out, p_in, p_out],
                                       [p_out, p_out, p_in]],
                              seed=0)
y = np.array([G.nodes[n]["block"] for n in G.nodes])      # node labels
N = G.number_of_nodes()
print(f"graph: {N} nodes, {G.number_of_edges()} edges, "
      f"{len(set(y))} classes")


# ───────────────────────────────────────────────────────────
# Adjacency, normalized
# ───────────────────────────────────────────────────────────
A = nx.to_numpy_array(G).astype(np.float32)
A_hat = A + np.eye(N)                       # add self-loops
deg = A_hat.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt    # symmetrically-normalized
A_norm = torch.from_numpy(A_norm)

# Initial features: just an identity matrix (no node features given).
# Each node learns its own embedding through layers.
H0 = torch.eye(N)


# ───────────────────────────────────────────────────────────
# 2-layer GCN (~10 lines)
# ───────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return A @ self.lin(x)              # propagate then project


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden)
        self.l2 = GCNLayer(hidden, n_classes)

    def forward(self, x, A):
        h = F.relu(self.l1(x, A))
        return self.l2(h, A)


model = GCN(in_dim=N, hidden=16, n_classes=3)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Semi-supervised training: only 5 labels per class are revealed
# ───────────────────────────────────────────────────────────
labels = torch.from_numpy(y)
train_mask = torch.zeros(N, dtype=torch.bool)
for c in range(3):
    cls_idx = np.where(y == c)[0][:5]       # first 5 of each class
    train_mask[cls_idx] = True

opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-4)

print(f"\nlabeled nodes: {train_mask.sum().item()} / {N}")
for epoch in range(200):
    opt.zero_grad()
    logits = model(H0, A_norm)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    opt.step()
    if epoch % 25 == 0 or epoch == 199:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            train_acc = (preds[train_mask] == labels[train_mask]).float().mean().item()
            test_acc  = (preds[~train_mask] == labels[~train_mask]).float().mean().item()
        print(f"epoch {epoch:>3}  loss={loss.item():.3f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")


# ───────────────────────────────────────────────────────────
# Visualize the graph + the predictions
# ───────────────────────────────────────────────────────────
with torch.no_grad():
    preds = model(H0, A_norm).argmax(dim=1).numpy()
pos = nx.spring_layout(G, seed=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nx.draw(G, pos, ax=axes[0], node_color=y, cmap="tab10", node_size=70,
        edge_color="lightgrey", with_labels=False)
axes[0].set_title("True classes")

nx.draw(G, pos, ax=axes[1], node_color=preds, cmap="tab10", node_size=70,
        edge_color="lightgrey", with_labels=False)
axes[1].set_title("GCN predictions")
fig.tight_layout()

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "06_gcn.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '06_gcn.png'}")


# ───────────────────────────────────────────────────────────
# Beyond GCN
# ───────────────────────────────────────────────────────────
# • GAT (Graph Attention) — replace fixed adjacency weights with attention.
# • GraphSAGE — sample neighbours; scales to large graphs.
# • R-GCN — relational graphs (multiple edge types).
# • GIN — provably more expressive than GCN.
#
# Real work: don't roll your own. Use `torch_geometric` or `dgl`. They
# implement these layers with dataloaders, sampling, and benchmarks.


# ───────────────────────────────────────────────────────────
# When NOT to use a GNN
# ───────────────────────────────────────────────────────────
# • Most "tabular" problems with no real graph structure: gradient boosting
#   wins almost always.
# • Pure text / image: transformers / CNNs are the right tool.
# • If the graph is just a binary "interacts with" — a two-tower model with
#   a flat negative-sampling loss often matches a GNN at much lower cost.
