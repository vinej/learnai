"""
05 — Metrics and structured logging

Three pillars of observability:

  LOGS    — events with context. "User X did Y at time T." Free-form text
             but PARSEABLE (structured / JSON).
  METRICS — numeric time series. "Requests per second", "P95 latency",
             "error rate". Cheap to store, fast to query.
  TRACES  — causal chain of operations across services. "Request A → DB
             query B → LLM call C". Module 4.4.07.

This file shows logs + metrics. Tracing for LLM-specific work is in 4.5.06.

Run: python 05_metrics_logging.py
"""

import logging
import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


# ───────────────────────────────────────────────────────────
# Structured logging — JSON to stdout
# ───────────────────────────────────────────────────────────
# Prefer JSON over plaintext. Aggregators (Datadog, Loki, ELK) parse JSON
# automatically; you can filter by any field.
import json


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything extra= keyword on log calls becomes a top-level field.
        for k, v in record.__dict__.items():
            if k not in {"args", "msg", "levelname", "levelno", "pathname",
                          "filename", "module", "exc_info", "exc_text",
                          "stack_info", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "name", "message"}:
                payload[k] = v
        return json.dumps(payload)


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
root.handlers = [handler]

log = logging.getLogger("predict")
log.info("model loaded", extra={"model_version": "0.1.0", "params": 12345})
log.warning("slow prediction", extra={"latency_ms": 412, "user_id": "u_99"})


# ───────────────────────────────────────────────────────────
# Metrics — Prometheus client
# ───────────────────────────────────────────────────────────
# Three core metric types:
#   COUNTER   — only goes up.    "requests_total"
#   GAUGE     — up and down.     "queue_depth"
#   HISTOGRAM — bucketed distribution. "request_latency_seconds"
registry = CollectorRegistry()

REQUESTS = Counter(
    "predict_requests_total",
    "Total prediction requests",
    labelnames=("status",),
    registry=registry,
)
LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction latency",
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=registry,
)


# ───────────────────────────────────────────────────────────
# Instrument a fake handler
# ───────────────────────────────────────────────────────────
def predict(features):
    with LATENCY.time():
        time.sleep(0.02)               # pretend work
        REQUESTS.labels(status="ok").inc()
        return {"score": 0.93}


for _ in range(50):
    predict({})
predict({})  # one "error"-flavored increment for demo:
REQUESTS.labels(status="error").inc()


# ───────────────────────────────────────────────────────────
# Print Prometheus exposition format (what /metrics returns)
# ───────────────────────────────────────────────────────────
print("\n=== /metrics endpoint output ===")
print(generate_latest(registry).decode())


# ───────────────────────────────────────────────────────────
# Wiring this into FastAPI
# ───────────────────────────────────────────────────────────
WIRING = """\
# In your FastAPI app:
from prometheus_client import make_asgi_app, Counter, Histogram

REQUESTS = Counter("predict_requests_total", "...", ["status"])
LATENCY  = Histogram("predict_latency_seconds", "...")

app.mount("/metrics", make_asgi_app())   # exposes the Prometheus format

@app.post("/predict")
def predict(features: HouseFeatures):
    with LATENCY.time():
        try:
            ...
            REQUESTS.labels(status="ok").inc()
        except Exception:
            REQUESTS.labels(status="error").inc()
            raise
"""
print("=== wiring into FastAPI ===")
print(WIRING)


# ───────────────────────────────────────────────────────────
# What to actually graph
# ───────────────────────────────────────────────────────────
DASHBOARDS = """\
THE FOUR GOLDEN SIGNALS (Google SRE)
  - Latency: P50, P95, P99 (histograms).
  - Traffic: requests per second.
  - Errors: ratio of 5xx and 4xx.
  - Saturation: CPU, memory, queue depth.

LLM-SPECIFIC ADDITIONS
  - Tokens per request (input vs output).
  - Cost per request — derive from tokens × pricing.
  - Cache hit rate (file 4.1.06) — how often did caching kick in.
  - Time to first token (TTFT) — user-perceived latency on streams.
  - Tokens per second during decode.
  - Hallucination rate (file 4.5.08) — sampled judge, slower cadence.

ALERTING (PagerDuty / OpsGenie)
  - Error rate > 1% for 5 min  → page.
  - P95 latency > 2s for 5 min → page.
  - Cost burn rate > budget    → email a human.
"""
print(DASHBOARDS)


# ───────────────────────────────────────────────────────────
# Best practices
# ───────────────────────────────────────────────────────────
BEST = """\
- Pick LOW-CARDINALITY label values. user_id is BAD; status_code is fine.
  Each unique label combination is a separate time series; thousands of
  user_ids → millions of series → out-of-memory Prometheus.

- Use HISTOGRAM (with sensible buckets) for latency. Avoid raw GAUGE.

- Add a request_id to EVERY log line. Propagate it across services as a
  header. Use it to correlate logs, metrics labels, and traces.

- Don't log full LLM inputs / outputs at INFO — they balloon the log
  store and may contain PII. Log content lengths + hashes; keep full
  text in a separate secured trace store.

- Correlate cost: log token usage from EVERY API call. Aggregate to
  Grafana for $/min, $/user, $/feature dashboards.
"""
print(BEST)
