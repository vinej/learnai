"""
Exercise 2 — Add Prometheus metrics + a tiny scraping demo

Wraps Exercise 1's predictor with metrics:
  • predict_requests_total{status}
  • predict_latency_seconds (histogram)
  • predict_errors_total

Then runs an in-process scraper that hits the API and prints the metrics
output. In real life you'd point Prometheus at /metrics and visualize in
Grafana — the prometheus.yml sample at the bottom shows how.

Run:
  python exercises/02_metrics_dashboard.py train
  python exercises/02_metrics_dashboard.py serve
  # in another terminal:
  python exercises/02_metrics_dashboard.py demo
"""

import sys
import time
from pathlib import Path

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, make_asgi_app
from pydantic import BaseModel, Field


HERE = Path(__file__).parent
MODEL_PATH = HERE / "scratch" / "model.joblib"


# ───────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────
REQUESTS = Counter(
    "predict_requests_total",
    "Total prediction requests",
    labelnames=("status",),
)
LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ───────────────────────────────────────────────────────────
# Schemas
# ───────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    bedrooms:  int   = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=20)
    sqft:      int   = Field(gt=0, le=20_000)
    lot_size:  int   = Field(ge=0, le=1_000_000)


class Prediction(BaseModel):
    predicted_price: float


def make_app() -> FastAPI:
    if not MODEL_PATH.exists():
        raise RuntimeError("run `train` first")

    app = FastAPI(title="House price predictor (metrics)", version="0.2.0")
    app.state.model = joblib.load(MODEL_PATH)

    # Mount /metrics for Prometheus
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=Prediction)
    def predict(features: HouseFeatures):
        with LATENCY.time():
            try:
                x = [[features.bedrooms, features.bathrooms,
                      features.sqft, features.lot_size]]
                y = float(app.state.model.predict(x)[0])
                REQUESTS.labels(status="ok").inc()
                return Prediction(predicted_price=round(y, 2))
            except Exception as e:
                REQUESTS.labels(status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))

    return app


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────
def train():
    """Reuse Exercise 1's training code by importing it."""
    sys.path.insert(0, str(HERE))
    from importlib import import_module
    import_module("01_fastapi_predict").train()


def serve():
    import uvicorn
    print("serving on :8000")
    print(" - /predict   (POST)")
    print(" - /metrics   (Prometheus exposition format)")
    print(" - /health")
    uvicorn.run(make_app(), host="127.0.0.1", port=8000)


def demo():
    base = "http://127.0.0.1:8000"
    print("driving 30 requests through the API ...")
    for i in range(30):
        # Mix valid + invalid to exercise the error counter
        payload = {"bedrooms": 3, "bathrooms": 2, "sqft": 1800, "lot_size": 7000}
        if i % 7 == 0:
            payload["sqft"] = 0   # invalid → 422 (NOT counted as 'error' below;
                                   # 422 is rejected by Pydantic before our handler.)
        try:
            httpx.post(f"{base}/predict", json=payload, timeout=5)
        except Exception:
            pass

    metrics = httpx.get(f"{base}/metrics", timeout=5).text
    # Extract the lines we care about
    interesting = [
        line for line in metrics.splitlines()
        if line.startswith("predict_requests_total")
        or line.startswith("predict_latency_seconds_count")
        or line.startswith("predict_latency_seconds_sum")
    ]
    print("\n=== /metrics excerpt ===")
    for line in interesting:
        print(line)

    # ── Sanity ──
    assert any('status="ok"' in line for line in interesting), \
        "expected predict_requests_total{status=\"ok\"} present"
    print("\nmetrics endpoint is wired up ✅")


# ───────────────────────────────────────────────────────────
# Reference: prometheus.yml (just for reading)
# ───────────────────────────────────────────────────────────
PROMETHEUS_YML = """\
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: house-predictor
    static_configs:
      - targets: ['host.docker.internal:8000']    # local API on host
    metrics_path: /metrics
"""


# ───────────────────────────────────────────────────────────
# Reference: a Grafana panel JSON snippet
# ───────────────────────────────────────────────────────────
PANEL_HINT = """\
USEFUL GRAFANA QUERIES (PromQL)
  - QPS by status:
        sum by (status) (rate(predict_requests_total[1m]))
  - P95 latency:
        histogram_quantile(0.95, sum by (le) (rate(predict_latency_seconds_bucket[5m])))
  - Error rate:
        sum(rate(predict_requests_total{status="error"}[1m]))
        / sum(rate(predict_requests_total[1m]))
"""


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Reference prometheus.yml:")
        print(PROMETHEUS_YML)
        print(PANEL_HINT)
        sys.exit(1)
    action = sys.argv[1]
    if action == "train":
        train()
    elif action == "serve":
        serve()
    elif action == "demo":
        demo()
    else:
        print(f"unknown: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
