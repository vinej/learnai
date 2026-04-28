"""
03 — Generative Adversarial Networks (GANs)

Two networks playing a game:

  GENERATOR G(z)        — takes random noise z, produces fake samples.
  DISCRIMINATOR D(x)    — takes a sample, predicts probability "real".

  D's loss: classify real vs fake correctly.
  G's loss: fool D into thinking its samples are real.

At equilibrium, G's outputs are indistinguishable from real data.

Practical issues you'll meet:
  MODE COLLAPSE   — G produces only a few patterns that fool D.
  INSTABILITY     — losses oscillate; tweaking one breaks the other.
  WGAN, WGAN-GP, spectral norm — popular fixes.

This file demonstrates a TINY GAN that learns to generate samples from a
1-D Gaussian mixture. CPU-friendly, end-to-end runnable in seconds.

Run: python 03_gans.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# True data: a mixture of two Gaussians
# ───────────────────────────────────────────────────────────
def sample_real(n: int) -> torch.Tensor:
    half = n // 2
    return torch.cat([
        torch.randn(half) * 0.5 + 2.0,        # bump at +2
        torch.randn(n - half) * 0.5 - 2.0,    # bump at -2
    ]).unsqueeze(1)


# ───────────────────────────────────────────────────────────
# G and D
# ───────────────────────────────────────────────────────────
LATENT_DIM = 8

G = nn.Sequential(
    nn.Linear(LATENT_DIM, 64), nn.ReLU(),
    nn.Linear(64, 64),         nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

D = nn.Sequential(
    nn.Linear(1, 64),  nn.LeakyReLU(0.2),
    nn.Linear(64, 64), nn.LeakyReLU(0.2),
    nn.Linear(64, 1),                  # logits — use BCEWithLogitsLoss
).to(device)


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

BATCH = 256
STEPS = 2000

losses_g, losses_d = [], []
for step in range(STEPS):
    # === Train D ===
    real = sample_real(BATCH).to(device)
    z = torch.randn(BATCH, LATENT_DIM, device=device)
    fake = G(z).detach()                  # detach: don't backprop through G

    d_real = D(real)
    d_fake = D(fake)
    loss_d = bce(d_real, torch.ones_like(d_real)) + \
             bce(d_fake, torch.zeros_like(d_fake))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    # === Train G ===
    z = torch.randn(BATCH, LATENT_DIM, device=device)
    fake = G(z)
    d_fake = D(fake)
    # Goal: D thinks the fake is REAL (label = 1).
    loss_g = bce(d_fake, torch.ones_like(d_fake))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    losses_g.append(loss_g.item())
    losses_d.append(loss_d.item())
    if step % 250 == 0:
        print(f"step {step:>4}: D loss={loss_d.item():.3f}  G loss={loss_g.item():.3f}")


# ───────────────────────────────────────────────────────────
# Inspect: distribution of G's outputs
# ───────────────────────────────────────────────────────────
with torch.no_grad():
    z = torch.randn(5000, LATENT_DIM, device=device)
    fake = G(z).cpu().squeeze().numpy()
real_np = sample_real(5000).squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(real_np, bins=80, alpha=0.6, label="real")
axes[0].hist(fake,    bins=80, alpha=0.6, label="generated")
axes[0].set_title("Real vs generated distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(losses_d, label="D loss")
axes[1].plot(losses_g, label="G loss")
axes[1].set_title("GAN training losses")
axes[1].set_xlabel("step")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "03_gan.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '03_gan.png'}")


# ───────────────────────────────────────────────────────────
# What you should observe
# ───────────────────────────────────────────────────────────
# • G's distribution should approximate the two-bump real distribution.
# • If D crushes G early (D loss → 0), G gets no useful gradient — that's
#   the "vanishing gradient" failure of vanilla GANs.
# • If you train too long, the losses can oscillate. Real GAN training is
#   notoriously sensitive to hyperparameters.


# ───────────────────────────────────────────────────────────
# Modern GAN family (image generation)
# ───────────────────────────────────────────────────────────
# • DCGAN          — first stable GAN for images. Conv generator/disc.
# • WGAN, WGAN-GP  — Wasserstein loss + gradient penalty. Way more stable.
# • Progressive    — train at low res first, gradually add layers.
# • StyleGAN(2/3)  — style-based generator. SOTA on faces, art, etc.
# • Pix2Pix, CycleGAN — image-to-image translation.
#
# Today, for high-quality image generation, DIFFUSION MODELS (next file)
# usually beat GANs. GANs remain useful for fast inference (1 forward pass)
# and specialized applications like super-resolution and style transfer.
