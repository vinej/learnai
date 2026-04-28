"""
01 — Serving an ML model with FastAPI

FastAPI is the de-facto Python framework for ML APIs:
  • Pydantic-typed requests and responses out of the box.
  • Auto-generated OpenAPI docs at /docs and /redoc.
  • Async-friendly (handle slow downstream APIs without blocking).
  • Tiny dependencies, fast.

The minimum viable pattern:

  1. Train (or load) the model.
  2. Define request and response Pydantic models.
  3. Wrap predict() in a FastAPI route.
  4. Run with uvicorn.

This file shows the pattern. Exercise 1 has a runnable end-to-end version.

Run: python 01_fastapi_serving.py
"""

# ───────────────────────────────────────────────────────────
# A complete minimal serving file
# ───────────────────────────────────────────────────────────
EXAMPLE = """\
# server.py
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ── Models load once at startup, then serve many requests ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("model.joblib")
    app.state.feature_names = ["bedrooms", "bathrooms", "sqft", "lot_size"]
    yield
    # cleanup happens after shutdown


app = FastAPI(title="House price predictor", version="0.1.0", lifespan=lifespan)


# ── Schemas ──
class HouseFeatures(BaseModel):
    bedrooms:   int   = Field(ge=0, le=20)
    bathrooms:  float = Field(ge=0, le=20)
    sqft:       int   = Field(gt=0, le=20000)
    lot_size:   int   = Field(ge=0, le=1_000_000)


class Prediction(BaseModel):
    predicted_price: float
    model_version:   str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures) -> Prediction:
    try:
        x = [[features.bedrooms, features.bathrooms,
              features.sqft, features.lot_size]]
        y = float(app.state.model.predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Prediction(predicted_price=y, model_version="0.1.0")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

print("=== a complete FastAPI ML server ===")
print(EXAMPLE)


# ───────────────────────────────────────────────────────────
# Why each piece is there
# ───────────────────────────────────────────────────────────
RATIONALE = """\
LIFESPAN
  Load the model ONCE at startup. Don't load per-request — disks are slow.
  Use `app.state` to share objects across requests.

PYDANTIC SCHEMAS
  Validate input AT THE BOUNDARY. Catch bad inputs before they hit your
  model. Constraints (ge, le, pattern) are documented automatically.

RESPONSE MODELS
  `response_model=...` does TWO things: it documents the response shape,
  AND filters extra keys before returning to the client. Belt-and-suspenders
  privacy.

EXCEPTION HANDLING
  Wrap risky calls in try/except, raise HTTPException with a clean status.
  FastAPI's default 422 covers Pydantic validation; 500 is your fallback.

HEALTH ENDPOINT
  Required by every container orchestrator (k8s liveness/readiness probes).
"""
print(RATIONALE)


# ───────────────────────────────────────────────────────────
# Running it in production
# ───────────────────────────────────────────────────────────
PRODUCTION = """\
DEV
  uvicorn server:app --reload --port 8000

PRODUCTION (Linux/macOS, gunicorn + uvicorn workers)
  gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

  - Workers (-w): roughly 2 × #CPU cores. More for IO-bound, fewer for CPU.
  - Each worker is its own process; the model is loaded in each.
  - For LARGE models (LLMs), prefer one worker + async + pipeline parallel
    (e.g. vLLM) — copying a 4GB model 4× is wasteful.

GUNICORN ALTERNATIVES
  - hypercorn — supports HTTP/2 + WebSockets natively.
  - uvicorn --workers — same idea, simpler. Fine for many cases.

BEHIND A REVERSE PROXY
  - nginx / Caddy / Traefik handles TLS, gzip, request limits.
  - Your gunicorn binds to 127.0.0.1; the proxy is the public surface.
"""
print(PRODUCTION)


# ───────────────────────────────────────────────────────────
# Other ML serving frameworks
# ───────────────────────────────────────────────────────────
FRAMEWORKS = """\
FastAPI            our default. General-purpose, ML-friendly, easy to test.
BentoML            ML-specific: model registry, batching, multi-model deploys.
Ray Serve          when you need multi-node and pipeline parallelism.
Triton (NVIDIA)    GPU-optimized; supports many backends (TF, PyTorch, ONNX).
TorchServe         PyTorch-native; loses to vLLM for LLMs.
LitServe           lightweight, async-friendly alternative to FastAPI.

For LLMs specifically, you usually skip "general" frameworks and use:
  vLLM, TGI, TensorRT-LLM, Ollama (file 4.3.07).
"""
print(FRAMEWORKS)


# ───────────────────────────────────────────────────────────
# Common gotchas
# ───────────────────────────────────────────────────────────
# • Forgetting to mount /metrics or /health → orchestrators can't probe.
# • Loading the model lazily inside the request handler → first request is slow.
# • Not setting a request timeout → one slow client takes down the worker.
# • Using `def` (sync) for an LLM API call → blocks the event loop.
#    Use `async def` and the `anthropic.AsyncAnthropic` client.
# • Returning datetime.datetime() without a serializer → silent 500.
