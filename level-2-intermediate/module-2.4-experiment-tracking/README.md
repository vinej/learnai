# Module 2.4 — Experiment Tracking & Reproducibility

**Level:** 2 — Intermediate
**Estimated time:** 1 week

## Goal
Make every experiment reproducible and comparable. Without this, ML work becomes an unmanageable mess after ~10 runs.

## Topics
- Why tracking matters (the "I had a 0.92 model last week and can't recreate it" problem)
- **MLflow:** experiments, runs, params, metrics, artifacts, model registry
- **Weights & Biases:** runs, sweeps, dashboards, reports
- Versioning data with **DVC** or **lakeFS**
- Random seeds: `random`, `numpy`, `torch` — and why they're not enough alone
- Environment locks: `pip freeze`, `pip-tools`, `uv`, `conda` env files
- Model cards & dataset cards (Hugging Face)

## Exercises
1. Take a previous Module 2.2 project and instrument it with MLflow — log params, metrics, and the model artifact.
2. Run a hyperparameter sweep with W&B Sweeps; compare runs in the dashboard.
3. Version a dataset with DVC and tie a model run to a specific data version.
4. Lock your project's Python deps with `uv` or `pip-tools` and rebuild the env from scratch.

## Capstone (Level 2)
Pick a Kaggle competition (active or past). Deliver:
- A clean, well-engineered feature pipeline
- At least 3 trained model variants, all logged in MLflow or W&B
- Cross-validated results with a baseline beat
- A short write-up of what worked and what didn't

## Resources
- MLflow docs: https://mlflow.org/docs/latest/index.html
- W&B docs: https://docs.wandb.ai/
- DVC: https://dvc.org/

## Checkpoint
Anyone (including future-you) can pull your repo, recreate the environment, rerun your best experiment, and reproduce the metrics within noise.
