"""
Exercise 2 — Train a VAE with 2-D latent on MNIST and visualize

The script:
  1. Trains a 2-D-latent VAE on MNIST (3 epochs).
  2. Plots a 2-D scatter of test-set encodings, colored by digit class.
  3. Decodes a 12×12 grid sweep across the latent space.

Asserts that classes are visibly separated: the average WITHIN-class distance
is smaller than the average BETWEEN-class distance in the latent.

Run: python exercises/02_vae_mnist.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transforms.ToTensor())
test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=2048, shuffle=False)


# ───────────────────────────────────────────────────────────
# VAE
# ───────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),         nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kl) / x.size(0)


model = VAE(latent_dim=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)


# ───────────────────────────────────────────────────────────
# Train (3 epochs)
# ───────────────────────────────────────────────────────────
print("training VAE ...")
for epoch in range(3):
    running, n = 0.0, 0
    for x, _ in train_loader:
        x = x.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    print(f"  epoch {epoch + 1}: loss/sample = {running / n:.2f}")


# ───────────────────────────────────────────────────────────
# Inspect the latent space
# ───────────────────────────────────────────────────────────
model.eval()
xs, ys = next(iter(test_loader))
with torch.no_grad():
    mu, _ = model.encode(xs.to(device))
zs = mu.cpu().numpy()
ys = ys.numpy()


# Compute within- vs between-class distances
def class_distance_stats(z, y):
    classes = np.unique(y)
    centroids = np.array([z[y == c].mean(axis=0) for c in classes])
    within = np.mean([np.linalg.norm(z[y == c] - centroids[i], axis=1).mean()
                      for i, c in enumerate(classes)])
    between = np.mean([np.linalg.norm(centroids[i] - centroids[j])
                       for i in range(len(classes)) for j in range(i + 1, len(classes))])
    return within, between


within, between = class_distance_stats(zs, ys)
print(f"\nlatent within-class avg dist:  {within:.3f}")
print(f"latent between-class avg dist: {between:.3f}")
ratio = between / within
print(f"between/within ratio: {ratio:.2f}  (>1 = separated)")


# ───────────────────────────────────────────────────────────
# Plots
# ───────────────────────────────────────────────────────────
# (a) latent scatter colored by digit
# (b) decoded grid
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
sc = axes[0].scatter(zs[:, 0], zs[:, 1], c=ys, cmap="tab10", s=4, alpha=0.6)
axes[0].set_title("VAE latent (colored by digit)")
plt.colorbar(sc, ax=axes[0])

grid = 12
xs_g = np.linspace(-3, 3, grid)
ys_g = np.linspace(-3, 3, grid)
canvas = np.zeros((28 * grid, 28 * grid))
with torch.no_grad():
    for i, gy in enumerate(ys_g):
        for j, gx in enumerate(xs_g):
            z = torch.tensor([[gx, gy]], dtype=torch.float32, device=device)
            img = model.decode(z).cpu().squeeze().numpy()
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
axes[1].imshow(canvas, cmap="gray")
axes[1].set_title("Decoded grid sweep")
axes[1].axis("off")

fig.tight_layout()
out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "exercise_02_vae.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / 'exercise_02_vae.png'}")


assert ratio > 1.0, (
    f"latent should separate classes (between/within > 1), got {ratio:.2f}. "
    "Try training more epochs or increasing the encoder capacity."
)
print("VAE latent separates classes ✅")
