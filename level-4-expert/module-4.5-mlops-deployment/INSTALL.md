# Setup — Module 4.5 (MLOps & Deployment)

This module covers what happens AFTER the model is trained: serving, containerizing, CI/CD, observability, reliability.

The lessons mix runnable Python with config files (Dockerfile, GitHub Actions, prometheus.yml). You don't need a Kubernetes cluster to learn the patterns.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

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

| Package              | Purpose                                          |
|----------------------|--------------------------------------------------|
| `fastapi`, `uvicorn` | The API server                                   |
| `pydantic`           | Request/response schemas                         |
| `scikit-learn`, `joblib` | Tiny model to serve                          |
| `prometheus-client`  | Metrics                                          |
| `httpx`              | Async HTTP client (load test)                    |
| `anthropic`          | LLM observability examples                       |

### Optional

| Tool       | Used in                                   |
|------------|-------------------------------------------|
| Docker     | `03_docker.py`, exercise 1 stretch goal    |
| GitHub     | `04_cicd.py`, exercise 3                  |
| Langfuse   | `06_llm_observability.py`                 |
| `locust`   | exercise 5 stretch goal                   |

You don't need Docker / Kubernetes installed to LEARN the patterns. The Dockerfile and Helm chart files are included as reference templates.

## 4. Run the lessons

```bash
python 01_fastapi_serving.py        # offline; reading code
python 02_streaming.py
python 03_docker.py                  # text reading + Dockerfile reference
python 04_cicd.py                    # text reading + workflow reference
python 05_metrics_logging.py
python 06_llm_observability.py       # API
python 07_reliability.py
python 08_monitoring_quality.py
```

## 5. Run the exercises

```bash
# Exercise 1: train + serve a model
python exercises/01_fastapi_predict.py train          # writes model.joblib
python exercises/01_fastapi_predict.py serve          # http://localhost:8000

# In another terminal:
python exercises/01_fastapi_predict.py demo

# Exercise 2: metrics
python exercises/02_metrics_dashboard.py serve        # http://localhost:8000/metrics

# Exercise 3: CI workflow (just inspect / commit to a repo)
cat exercises/03_ci_workflow.yml

# Exercise 4: LLM tracing
python exercises/04_observability.py                  # writes JSON traces

# Exercise 5: async load test
python exercises/05_load_test.py
```

## A note on production parity

The patterns in this module are real but the deployments are MOCK (localhost, in-memory, single-process). To take them to prod:

- **Serving:** add gunicorn workers / vLLM behind a load balancer.
- **Containers:** push to a registry, deploy to Kubernetes / Cloud Run / Lambda.
- **Observability:** ship metrics to Prometheus + Grafana, logs to Loki / Datadog, traces to Jaeger / Honeycomb / Langfuse.
- **CI/CD:** the workflow in exercise 3 is a starting point.

Module 4.6 (Responsible & Scalable AI) covers the safety/security axis.
