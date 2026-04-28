"""
02 — Variational Autoencoders (VAE)

A standard autoencoder's latent space is "spiky" — far from any training z,
the decoder produces garbage. So you can't easily SAMPLE new images.

A VAE fixes this by:
  1. The encoder outputs (μ, log σ²) of a Gaussian, not a point.
  2. We sample z ~ N(μ, σ²), decode it.
  3. We add a KL penalty pushing the encoder toward N(0, I).
  Result: a smooth, well-behaved latent space you CAN sample from.

LOSS = reconstruction (BCE or MSE) + β * KL(q(z|x) || N(0, I))

The "REPARAMETERIZATION TRICK" — z = μ + σ * ε, ε ~ N(0, I) — keeps the
backprop path differentiable through the random sample.

Run: python 02_vae.py
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

DATA_DIR = Path(__file__).parent / "data"
mnist = datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=128, shuffle=True, num_workers=0)


# ───────────────────────────────────────────────────────────
# VAE with 2-D latent for visualization
# ───────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ───────────────────────────────────────────────────────────
# VAE loss: reconstruction + KL
# ───────────────────────────────────────────────────────────
def vae_loss(recon, x, mu, logvar):
    # BCE — pixel-wise; sum across image (not mean — keeps things in scale)
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL between N(mu, sigma) and N(0, 1), closed form:
    #   -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kl) / x.size(0), bce.item() / x.size(0), kl.item() / x.size(0)


# ───────────────────────────────────────────────────────────
# Train (quick: 3 epochs)
# ───────────────────────────────────────────────────────────
model = VAE(latent_dim=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("training VAE ...")
for epoch in range(3):
    running, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, bce, kl = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    print(f"  epoch {epoch + 1}: total/sample={running / n:.2f}  bce={bce:.2f}  kl={kl:.2f}")


# ───────────────────────────────────────────────────────────
# Visualize the 2-D latent space
# ───────────────────────────────────────────────────────────
model.eval()

# (a) plot a sample of training points colored by their digit class
sample_loader = DataLoader(mnist, batch_size=2000, shuffle=False)
xs, ys = next(iter(sample_loader))
with torch.no_grad():
    mu, _ = model.encode(xs.to(device))
zs = mu.cpu().numpy()

# (b) decode a grid of points across the latent space — generative sweep
grid = 12
xx = np.linspace(-3, 3, grid)
yy = np.linspace(-3, 3, grid)
canvas = np.zeros((28 * grid, 28 * grid))
with torch.no_grad():
    for i, gy in enumerate(yy):
        for j, gx in enumerate(xx):
            z = torch.tensor([[gx, gy]], dtype=torch.float32, device=device)
            img = model.decode(z).cpu().squeeze().numpy()
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sc = axes[0].scatter(zs[:, 0], zs[:, 1], c=ys.numpy(), cmap="tab10", s=4, alpha=0.7)
axes[0].set_title("VAE latent space (colored by digit class)")
plt.colorbar(sc, ax=axes[0], label="digit")
axes[0].grid(True, alpha=0.3)
axes[1].imshow(canvas, cmap="gray")
axes[1].set_title("Decoded 12×12 grid of latent points")
axes[1].axis("off")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "02_vae.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '02_vae.png'}")


# ───────────────────────────────────────────────────────────
# Sample novel digits — actually generative
# ───────────────────────────────────────────────────────────
with torch.no_grad():
    z = torch.randn(8, 2, device=device)
    samples = model.decode(z).cpu().squeeze().numpy()
print(f"\nsampled 8 novel digits from N(0, I): shape {samples.shape}")


# ───────────────────────────────────────────────────────────
# Notes on real-world VAEs
# ───────────────────────────────────────────────────────────
# • β-VAE: weight the KL term to control disentanglement.
# • VQ-VAE: discrete latent codes (used in Stable Diffusion's image encoder).
# • For high-quality image generation, diffusion models (next file) usually win.
# • For TABULAR data, VAEs are great for synthetic-data generation and
#   anomaly detection.
