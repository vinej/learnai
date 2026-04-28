"""
Exercise 3 — Matrix factorization for a synthetic ratings matrix

Generate a low-rank user-item ratings matrix, hide 80% of entries, and
recover them with matrix factorization. The script asserts the held-out
RMSE is significantly better than naive baselines.

Run: python exercises/03_matrix_factorization.py
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)
np.random.seed(0)


# ───────────────────────────────────────────────────────────
# Synthetic data: rank-K matrix + Gaussian noise, ratings 1..5
# ───────────────────────────────────────────────────────────
N_USERS, N_ITEMS, TRUE_K = 300, 150, 4
true_U = np.random.normal(0, 1, (N_USERS, TRUE_K))
true_V = np.random.normal(0, 1, (N_ITEMS, TRUE_K))
ratings = (true_U @ true_V.T) + np.random.normal(0, 0.2, (N_USERS, N_ITEMS))

# Squash to a 1..5 scale (standardize then map)
ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
ratings = ratings * 4 + 1                  # 1..5

# Hide 80%
mask = np.random.random(ratings.shape) < 0.2
print(f"observed ratings: {mask.sum()} / {ratings.size}")


# ───────────────────────────────────────────────────────────
# Convert observed entries to tensors
# ───────────────────────────────────────────────────────────
u_obs, i_obs = np.where(mask)
u_held, i_held = np.where(~mask)
y_obs  = torch.tensor(ratings[u_obs, i_obs],  dtype=torch.float32)
y_held = torch.tensor(ratings[u_held, i_held], dtype=torch.float32)
u_obs_t  = torch.tensor(u_obs,  dtype=torch.long)
i_obs_t  = torch.tensor(i_obs,  dtype=torch.long)
u_held_t = torch.tensor(u_held, dtype=torch.long)
i_held_t = torch.tensor(i_held, dtype=torch.long)


# ───────────────────────────────────────────────────────────
# MF model
# ───────────────────────────────────────────────────────────
class MFModel(nn.Module):
    def __init__(self, n_users, n_items, k, mean):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        self.user_b   = nn.Embedding(n_users, 1)
        self.item_b   = nn.Embedding(n_items, 1)
        self.global_b = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, u, i):
        return ((self.user_emb(u) * self.item_emb(i)).sum(-1)
                + self.user_b(u).squeeze() + self.item_b(i).squeeze()
                + self.global_b)


model = MFModel(N_USERS, N_ITEMS, k=8, mean=float(y_obs.mean()))
opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)
loss_fn = nn.MSELoss()

# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
for epoch in range(300):
    opt.zero_grad()
    pred = model(u_obs_t, i_obs_t)
    loss = loss_fn(pred, y_obs)
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            held = model(u_held_t, i_held_t)
            held_rmse = ((held - y_held) ** 2).mean().sqrt().item()
        print(f"epoch {epoch:>3}  train MSE {loss.item():.4f}  held RMSE {held_rmse:.4f}")


# ───────────────────────────────────────────────────────────
# Final eval against baselines
# ───────────────────────────────────────────────────────────
with torch.no_grad():
    pred_held = model(u_held_t, i_held_t)
mf_rmse = ((pred_held - y_held) ** 2).mean().sqrt().item()

# Baselines:
#   • global mean
#   • per-user mean
#   • per-item mean
g_mean = y_obs.mean().item()
global_rmse = ((g_mean - y_held) ** 2).mean().sqrt().item()

# per-user mean baseline
user_means = np.full(N_USERS, g_mean)
for u_id in range(N_USERS):
    m = (u_obs == u_id)
    if m.any():
        user_means[u_id] = ratings[u_obs[m], i_obs[m]].mean()
baseline_user = ((torch.tensor(user_means[u_held]) - y_held) ** 2).mean().sqrt().item()

print(f"\nfinal RMSE on held-out entries:")
print(f"  global mean baseline: {global_rmse:.3f}")
print(f"  per-user mean:        {baseline_user:.3f}")
print(f"  matrix factorization: {mf_rmse:.3f}")

assert mf_rmse < baseline_user, "MF should beat the per-user-mean baseline"
assert mf_rmse < global_rmse * 0.7, "MF should beat global mean by ≥ 30%"
print("matrix factorization ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Try k = 2, 4, 8, 32 — at what k do you start overfitting?
# • Add weight_decay sweeps. Stronger regularization helps in sparse regimes.
# • Implement BPR (Bayesian Personalized Ranking) loss for IMPLICIT feedback
#   (clicks, not ratings). It's the standard "you have only positives" loss.
# ─────────────────────────────────────────────────────────────────────────────
