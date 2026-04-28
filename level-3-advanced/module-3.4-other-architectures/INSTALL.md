# Setup — Module 3.4 (Other Architectures)

A whirlwind tour of architectures outside vanilla classification:
autoencoders, VAEs, GANs, diffusion intuition, recommenders, graph neural
networks, and time-series DL.

## 1. Python ≥ 3.11 + PyTorch

If you completed Module 3.1, you're set.

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package           | Purpose                                          |
|-------------------|--------------------------------------------------|
| `torch`, `torchvision` | The framework + MNIST                       |
| `numpy`, `matplotlib`  | Plots & arrays                              |
| `pandas`              | Tabular data for recommender                  |
| `scikit-learn`        | SVD baseline, synthetic data                  |
| `networkx`            | Graph visualization                           |

## 4. Run the lessons

```bash
python 01_autoencoders.py             # downloads MNIST ~10MB on first run
python 02_vae.py                      # uses MNIST cache
python 03_gans.py
python 04_diffusion_intuition.py
python 05_recommenders.py
python 06_graph_neural_networks.py
python 07_time_series_dl.py
```

Plotting scripts save PNGs to `figures/`.

## 5. Run the exercises

```bash
python exercises/01_denoising_ae_mnist.py   # ~3-5 min on CPU
python exercises/02_vae_mnist.py            # ~3-5 min on CPU
python exercises/03_matrix_factorization.py # ~30 seconds
python exercises/04_tiny_gcn.py             # ~30 seconds
```

## A note on scope

Each topic in this module deserves its own course. The goal here is to:
1. Understand the SHAPE of each architecture (inputs, outputs, loss, training).
2. Build a working tiny version so the math is concrete.
3. Know which professional libraries to reach for when you need the real thing.

For diffusion models, GANs at scale, real GNNs (use `torch_geometric`),
and production recommenders (use `Implicit` or `RecBole`), you'll want a
deeper dedicated course or paper-by-paper study.
