# Setup — Module 3.1 (Deep Learning Foundations)

This is the first deep-learning module. It uses **PyTorch** for the framework lessons and **NumPy** for the from-scratch backprop demo.

> **Disk warning:** PyTorch is large (~2 GB CPU build, ~5 GB+ with CUDA). Make sure you have room.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

## 2. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

## 3. Install PyTorch

The default `pip install torch` gets you a CPU build that works everywhere. For GPU support, follow the matrix at https://pytorch.org/get-started/locally/ — CUDA version, OS, and driver matter.

This module's exercises will train fine on CPU; MNIST (~5 minutes) and the bigger optimizer comparison (~2 minutes).

## 4. Install everything else

```bash
pip install -r requirements.txt
```

| Package         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `torch`         | The framework                                    |
| `torchvision`   | MNIST loader + image utilities                   |
| `numpy`, `matplotlib` | Math + plots                                |
| `tqdm`          | Progress bars                                    |
| `tensorboard`   | Optional: log training curves to a dashboard     |

## 5. Confirm CUDA / MPS (optional)

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
```

If both say `False`, you're on CPU — fine for this module. Module 3.5 (Training at Scale) digs into hardware.

## 6. Run the lessons

```bash
python 01_perceptron_to_mlp.py
python 02_activations_and_losses.py
python 03_backprop_numpy.py
python 04_pytorch_tensors.py
python 05_pytorch_autograd.py
python 06_nn_module.py
python 07_training_loop.py
python 08_optimizers.py
python 09_regularization.py
python 10_dataloaders_and_checkpoints.py
```

Plotting scripts save PNGs to `figures/`.

## 7. Run the exercises

```bash
python exercises/01_mlp_from_scratch_numpy.py
python exercises/02_mlp_pytorch.py
python exercises/03_mnist_classifier.py        # downloads ~10MB on first run
python exercises/04_optimizer_comparison.py
python exercises/05_regularization_effects.py
```

## TensorBoard (optional)

```bash
tensorboard --logdir runs
```

then open http://localhost:6006.

## Tip

Don't try to memorize the PyTorch API. Memorize the *training loop*:

```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
```

Every PyTorch program you'll ever write is a variation on those six lines.
