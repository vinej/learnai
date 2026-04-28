"""
Exercise 1 — Train a denoising autoencoder on MNIST

Add Gaussian noise to inputs; train the AE to reconstruct the clean image.
The script asserts the test reconstruction MSE on NOISY inputs is at least
30% better than just outputting the noisy input directly.

Run: python exercises/01_denoising_ae_mnist.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────
transform = transforms.ToTensor()
train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False)


# ───────────────────────────────────────────────────────────
# Model — convolutional denoising AE
# ───────────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 28→14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),    # 14→7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = ConvAE().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# ───────────────────────────────────────────────────────────
# Train: noisy input, clean target
# ───────────────────────────────────────────────────────────
NOISE_STD = 0.4
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running, n = 0.0, 0
    for x, _ in train_loader:
        x = x.to(device)
        noisy = (x + NOISE_STD * torch.randn_like(x)).clamp(0, 1)
        opt.zero_grad()
        recon = model(noisy)
        loss = loss_fn(recon, x)            # target is the CLEAN image
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    print(f"epoch {epoch + 1}/{EPOCHS}  train MSE {running / n:.5f}")


# ───────────────────────────────────────────────────────────
# Evaluate on test set
# ───────────────────────────────────────────────────────────
model.eval()
running_recon, running_noise, n = 0.0, 0.0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        noisy = (x + NOISE_STD * torch.randn_like(x)).clamp(0, 1)
        recon = model(noisy)
        running_recon += loss_fn(recon, x).item() * x.size(0)
        running_noise += loss_fn(noisy, x).item() * x.size(0)
        n += x.size(0)

recon_mse = running_recon / n
noise_mse = running_noise / n
improvement = (1 - recon_mse / noise_mse) * 100

print(f"\ntest MSE — noisy vs clean:    {noise_mse:.5f}")
print(f"test MSE — denoised vs clean: {recon_mse:.5f}")
print(f"improvement: {improvement:.1f}% over noisy baseline")


# ───────────────────────────────────────────────────────────
# Visualize a few clean / noisy / reconstructed triples
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 8, figsize=(13, 5))
sample, _ = next(iter(test_loader))
sample = sample[:8].to(device)
noisy = (sample + NOISE_STD * torch.randn_like(sample)).clamp(0, 1)
with torch.no_grad():
    recon = model(noisy)

for i in range(8):
    axes[0, i].imshow(sample[i].cpu().squeeze(), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(noisy[i].cpu().squeeze(),  cmap="gray"); axes[1, i].axis("off")
    axes[2, i].imshow(recon[i].cpu().squeeze(),  cmap="gray"); axes[2, i].axis("off")
fig.suptitle("Top: clean, Middle: noisy, Bottom: denoised")

out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "exercise_01_denoising_ae.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / 'exercise_01_denoising_ae.png'}")


assert improvement > 30.0, f"target ≥ 30%, got {improvement:.1f}%"
print("denoising AE works ✅")
