"""
03 — Containerization with Docker

Docker packages your app + dependencies into an image that runs the same
on your laptop, CI, and production. THE essential ML deployment tool.

Three rules:
  1. MULTI-STAGE BUILDS — separate "builder" (with compilers) from "runtime".
                          Smaller images, faster pulls, less attack surface.
  2. PIN VERSIONS — base image, Python, every pip dep. Reproducibility.
  3. SHIP NON-ROOT — your container shouldn't run as root.

Run: python 03_docker.py
"""

# ───────────────────────────────────────────────────────────
# A canonical Dockerfile for a FastAPI ML service
# ───────────────────────────────────────────────────────────
DOCKERFILE = """\
# ---- builder stage: install deps and compile any wheels ----
FROM python:3.11-slim-bookworm AS builder

# Avoid Python writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for any compiled wheels (numpy, scikit-learn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt


# ---- runtime stage: minimal final image ----
FROM python:3.11-slim-bookworm AS runtime

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Container-level health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \\
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
print("=== Dockerfile (multi-stage, non-root) ===")
print(DOCKERFILE)


# ───────────────────────────────────────────────────────────
# .dockerignore — keep the build context small
# ───────────────────────────────────────────────────────────
DOCKERIGNORE = """\
.git/
.venv/
__pycache__/
*.pyc
*.log
.pytest_cache/
.mypy_cache/
.ruff_cache/
node_modules/
.DS_Store
.env
.env.*
data/raw/
notebooks/
"""
print("\n=== .dockerignore ===")
print(DOCKERIGNORE)


# ───────────────────────────────────────────────────────────
# docker-compose.yml — local dev stack (API + monitoring)
# ───────────────────────────────────────────────────────────
COMPOSE = """\
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      start_period: 10s

  prometheus:
    image: prom/prometheus:latest
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: "Admin"
"""
print("\n=== docker-compose.yml ===")
print(COMPOSE)


# ───────────────────────────────────────────────────────────
# GPU images
# ───────────────────────────────────────────────────────────
GPU = """\
For GPU workloads (LLM inference, deep-learning training), use NVIDIA
base images:

  FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

For PyTorch:
  FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

For vLLM:
  FROM vllm/vllm-openai:latest

Run with the NVIDIA Container Runtime:
  docker run --gpus all my-image
  # in compose:
  deploy:
    resources:
      reservations:
        devices: [{capabilities: [gpu]}]

Watch the image size — full PyTorch+CUDA images are 5-10 GB. For inference,
prefer smaller bases (e.g. NVIDIA's Triton runtime image).
"""
print(GPU)


# ───────────────────────────────────────────────────────────
# Image hygiene checklist
# ───────────────────────────────────────────────────────────
CHECKLIST = """\
[ ] Multi-stage build      → smaller images, fewer compromises
[ ] Pinned base image SHA   → reproducible builds
[ ] requirements.txt pinned → exact same packages every time
[ ] Non-root user
[ ] HEALTHCHECK
[ ] .dockerignore           → don't ship .git, secrets, datasets
[ ] No secrets in layers    → use env vars, mounted secrets, NOT COPY
[ ] Vulnerability scan      → docker scout / trivy / grype on every push
[ ] Tagged with semantic version + git SHA
[ ] Pushed to a registry    → ghcr.io / Docker Hub / ECR / GAR
"""
print(CHECKLIST)


# ───────────────────────────────────────────────────────────
# Beyond Docker
# ───────────────────────────────────────────────────────────
BEYOND = """\
ORCHESTRATION
  Kubernetes — the standard for multi-container production. Steep learning curve.
  Cloud Run / Lambda — serverless, pay per request, autoscale to zero.
  Fly.io / Railway / Render — opinionated, "deploy from git" PaaS.

CI/CD
  Build & push the image on every commit.
  Deploy to a staging env automatically.
  Promote to prod with a tag / human approval. (See file 04.)

SECRETS
  - Local dev: .env files (NEVER commit) + python-dotenv.
  - CI: GitHub Actions secrets, OIDC for cloud auth.
  - Prod: Vault, AWS Secrets Manager, Doppler.
  - NEVER bake secrets into images.
"""
print(BEYOND)
