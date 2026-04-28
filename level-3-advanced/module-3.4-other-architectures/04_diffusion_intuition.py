"""
04 — Diffusion models (intuition)

Diffusion is the engine behind Stable Diffusion, DALL-E 3, Sora, and most
image / video / audio generators today.

THE IDEA, in one breath:
  • FORWARD process: gradually add Gaussian noise to a real image over T
    steps. After T steps, the result is pure noise.
  • REVERSE process: train a neural net to predict the noise added at each
    step, given the noisy image and the timestep. To generate, start from
    pure noise and iteratively REMOVE noise using the net's predictions.

You're training a DENOISER. Sampling = repeat denoising T times from random
noise. Slow (T = 50-1000 steps), but the quality is incredible.

This file:
  1. Visualizes the forward (noising) process on MNIST.
  2. Sketches the training objective without actually training (training a
     real diffusion model needs hours, even at MNIST scale).

Run: python 04_diffusion_intuition.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# The forward "noising" process
# ───────────────────────────────────────────────────────────
# At step t in [0, T], we have a noise schedule β_t (e.g. linear).
# Define α_t = 1 − β_t and ᾱ_t = ∏ α_s.
# Then  x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I).
# This is a closed-form jump from x_0 to x_t — no need to iterate.
T = 1000
beta = torch.linspace(1e-4, 2e-2, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


def q_sample(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t directly from x_0 at timestep t. Returns (x_t, noise)."""
    sqrt_ab = alpha_bar[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_one_minus * eps, eps


# ───────────────────────────────────────────────────────────
# Visualize a digit being noised at increasing timesteps
# ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
mnist = datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor())
img, _ = mnist[7]                        # one digit
img = img.unsqueeze(0)                   # (1, 1, 28, 28)
# Center to [-1, 1] — matches typical diffusion conventions
img = img * 2 - 1


checkpoints = [0, 50, 200, 500, 800, 999]
fig, axes = plt.subplots(1, len(checkpoints), figsize=(13, 2.5))
for ax, t in zip(axes, checkpoints):
    t_tensor = torch.tensor([t])
    x_t, _ = q_sample(img, t_tensor)
    ax.imshow(x_t.squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t={t}")
    ax.axis("off")
fig.suptitle("Forward (noising) process — at t=T it's pure noise")

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "04_diffusion_forward.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / '04_diffusion_forward.png'}")


# ───────────────────────────────────────────────────────────
# What the training loop looks like (sketch — not run)
# ───────────────────────────────────────────────────────────
# A diffusion model is typically a U-Net that takes (noisy_image, timestep)
# and predicts the noise.
#
#   for batch in loader:
#       x0 = batch.to(device)
#       t = torch.randint(0, T, (batch_size,))
#       x_t, eps = q_sample(x0, t)
#       eps_pred = model(x_t, t)                   # the U-Net's job
#       loss = F.mse_loss(eps_pred, eps)
#       loss.backward(); opt.step()
#
# That's it. The "deep" part is a U-Net (Module 3.2's segmentation idea
# applied here) with timestep embeddings injected via FiLM or addition.
# We don't train one here because real training takes hours even on GPU.


# ───────────────────────────────────────────────────────────
# Sampling (generation) at a glance
# ───────────────────────────────────────────────────────────
# Start with x_T ~ N(0, I). For t = T, T−1, ..., 1:
#   eps_pred = model(x_t, t)
#   compute the mean of the reverse Gaussian using closed-form math
#   x_{t-1} = mean + noise_term
# At t = 0, x_0 is your generated image.
#
# Faster samplers (DDIM, DPM-Solver) skip steps and finish in 20-50 forward
# passes instead of 1000.


# ───────────────────────────────────────────────────────────
# Why diffusion won
# ───────────────────────────────────────────────────────────
# • STABLE training compared to GANs (no two-network adversarial dance).
# • Massively better quality at scale.
# • Easy to CONDITION: feed the model a text embedding (CLIP / T5) at each
#   step → text-to-image. That's basically Stable Diffusion.
# • Latent diffusion: do all this in a smaller compressed latent space
#   (from a VAE) instead of pixel space. 10x faster, indistinguishable quality.


# ───────────────────────────────────────────────────────────
# Libraries to use in real work
# ───────────────────────────────────────────────────────────
# • Hugging Face `diffusers` — the standard for using/training diffusion.
#     pipe = StableDiffusionPipeline.from_pretrained(...)
#     image = pipe("a cat astronaut").images[0]
# • `accelerate` for distributed training.
# • For research-grade work, follow papers (DDPM, DDIM, Score SDE, EDM, Flow Matching).
