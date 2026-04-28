"""
01 — Autoencoders

An autoencoder learns to copy its input to its output, but FORCED through a
narrow bottleneck. The bottleneck representation must compress everything
useful in the input — that's the learned feature.

  ENCODER : x → z      (z is small, e.g. 32-D for a 784-D MNIST digit)
  DECODER : z → x̂     (reconstruction)
  Loss    : ||x − x̂||²

Variants:
  UNDERCOMPLETE (the default — bottleneck < input)
  DENOISING     — train with noisy inputs, clean targets. Learns robust feats.
  SPARSE        — penalize activations, encourage few neurons firing per input.

Run: python 01_autoencoders.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Data — small MNIST subset for speed
# ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

mnist = datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=128, shuffle=True, num_workers=0)


# ───────────────────────────────────────────────────────────
# Tiny autoencoder
# ───────────────────────────────────────────────────────────
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                        # (B, 784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),                        # output in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z).view(x.size(0), 1, 28, 28)
        return out, z


model = AutoEncoder(latent_dim=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


# ───────────────────────────────────────────────────────────
# Train — quick (3 epochs)
# ───────────────────────────────────────────────────────────
print("training autoencoder ...")
EPOCHS = 3
for epoch in range(EPOCHS):
    running, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    print(f"  epoch {epoch + 1}/{EPOCHS}  loss={running / n:.4f}")


# ───────────────────────────────────────────────────────────
# Visualize reconstructions
# ───────────────────────────────────────────────────────────
model.eval()
sample, _ = next(iter(loader))
sample = sample[:8].to(device)
with torch.no_grad():
    recon, _ = model(sample)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(sample[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("orig", rotation=0, labelpad=20)
axes[1, 0].set_ylabel("recon", rotation=0, labelpad=20)
fig.suptitle("Autoencoder reconstructions")

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "01_autoencoder.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '01_autoencoder.png'}")


# ───────────────────────────────────────────────────────────
# Denoising autoencoder — same architecture, noisy inputs
# ───────────────────────────────────────────────────────────
# Just add Gaussian noise to the input; the target stays the clean image.
def denoising_step(x, noise_std=0.4):
    noisy = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    return noisy


# ───────────────────────────────────────────────────────────
# Why autoencoders matter
# ───────────────────────────────────────────────────────────
# • Pretraining: train an AE on lots of unlabeled data, then use the encoder
#   as a feature extractor for downstream classification.
# • Anomaly detection: train on "normal" data, flag inputs with high
#   reconstruction error.
# • Compression: a 16-D bottleneck for a 784-D digit = 49× compression
#   (with some quality loss).
# • The DECODER alone is the foundation of generative models — see VAE next.
