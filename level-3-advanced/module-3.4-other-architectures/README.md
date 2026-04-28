# Module 3.4 — Other Architectures

**Level:** 3 — Advanced
**Estimated time:** 2 weeks

## Goal
Broaden your toolkit with architectures used outside vanilla classification/regression.

## Topics
### Generative
- **Autoencoders:** vanilla, denoising, sparse
- **Variational Autoencoders (VAEs):** latent space, KL divergence
- **GANs:** generator/discriminator, training instabilities, DCGAN, StyleGAN (conceptual)
- **Diffusion models:** forward/reverse process, DDPM intuition (conceptual; full impl is its own course)

### Recommenders
- Collaborative filtering (matrix factorization, SVD)
- Implicit feedback (ALS)
- Two-tower neural retrievers
- Sequential recommenders (SASRec, BERT4Rec)

### Graph Neural Networks (intro)
- Graphs as data: nodes, edges, features
- Message passing, GCN, GAT
- Use cases: fraud, drug discovery, social networks

### Time-series deep learning
- Temporal convolutions
- N-BEATS, Temporal Fusion Transformer
- When classical models (ARIMA, Prophet) still win

## Exercises
1. Train a denoising autoencoder on MNIST; reconstruct from noisy inputs.
2. Train a VAE; visualize the 2D latent space.
3. Implement matrix factorization for the MovieLens 100K dataset.
4. Build a tiny GCN on the Cora citation dataset.

## Resources
- Lilian Weng's blog (incredible deep dives on VAE/GAN/diffusion)
- PyTorch Geometric docs
- "Generative Deep Learning" — David Foster

## Checkpoint
You know which architecture family fits which kind of problem, and you've trained at least one generative model and one recommender.
