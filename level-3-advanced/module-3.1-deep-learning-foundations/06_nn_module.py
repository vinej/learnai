"""
06 — nn.Module: building reusable, parameterized blocks

`nn.Module` is the base class for every layer, model, and loss function in
PyTorch. It tracks the parameters (weights), exposes a `forward()` method,
and integrates with optimizers and serialization.

Three ways to build a model — pick by clarity, not by aesthetics:

  1. nn.Linear / nn.ReLU directly         (smallest)
  2. nn.Sequential([...])                  (when it's a straight pipeline)
  3. subclass nn.Module                    (when you need branches, residuals,
                                            multiple outputs, custom logic)

Run: python 06_nn_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────
# 1. Built-in single layers
# ───────────────────────────────────────────────────────────
linear = nn.Linear(in_features=4, out_features=2)
print("linear params:")
for name, p in linear.named_parameters():
    print(f"  {name}: shape {tuple(p.shape)}, requires_grad={p.requires_grad}")


# ───────────────────────────────────────────────────────────
# 2. Sequential: stack layers in order
# ───────────────────────────────────────────────────────────
mlp = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
x = torch.randn(8, 784)        # batch of 8 "flattened MNIST images"
y = mlp(x)
print(f"\nSequential output shape: {tuple(y.shape)}")
print(f"  total params: {sum(p.numel() for p in mlp.parameters())}")


# ───────────────────────────────────────────────────────────
# 3. Subclass nn.Module — full flexibility
# ───────────────────────────────────────────────────────────
class TwoHeadedNet(nn.Module):
    """A backbone with two output heads — common in multi-task learning."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, n_classes)
        self.regressor  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.classifier(h), self.regressor(h).squeeze(-1)


net = TwoHeadedNet(in_dim=20, hidden=64, n_classes=3)
sample = torch.randn(5, 20)
logits, regression = net(sample)
print(f"\nTwoHeadedNet outputs: logits {tuple(logits.shape)}, "
      f"regression {tuple(regression.shape)}")


# ───────────────────────────────────────────────────────────
# Parameters — `optimizer = ...(model.parameters(), lr=...)`
# ───────────────────────────────────────────────────────────
total = sum(p.numel() for p in net.parameters())
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\ntotal params: {total:,}, trainable: {trainable:,}")


# ───────────────────────────────────────────────────────────
# Modules of modules — composition
# ───────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Mini residual block — the trick behind ResNet (file 3.2 covers CNN ResNets)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)            # the SHORTCUT connection


class DeepNet(nn.Module):
    def __init__(self, in_dim, hidden, depth, n_classes):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.output = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = F.relu(self.input(x))
        h = self.blocks(h)
        return self.output(h)


deep = DeepNet(in_dim=10, hidden=64, depth=4, n_classes=2)
print(f"\nDeepNet:\n{deep}")
print(f"output shape on 16 inputs: {tuple(deep(torch.randn(16, 10)).shape)}")


# ───────────────────────────────────────────────────────────
# train() vs eval()
# ───────────────────────────────────────────────────────────
# Some layers behave differently during training vs inference (Dropout,
# BatchNorm). Always toggle the mode explicitly.
deep.eval()
print(f"\nmode: training={deep.training}")
deep.train()
print(f"mode: training={deep.training}")


# ───────────────────────────────────────────────────────────
# Tips
# ───────────────────────────────────────────────────────────
# • Always call super().__init__() FIRST in your __init__. Forgetting it is
#   a classic bug — Module won't track your sub-modules.
# • Anything stored as `self.something` that's a Module or Parameter is
#   tracked automatically. Plain Python lists are NOT — use nn.ModuleList /
#   nn.Sequential instead.
# • Nothing about nn.Module is magic — explore via `print(model)` and
#   `dict(model.named_parameters())`.
