# Module 4.5 — MLOps & Deployment

**Level:** 4 — Expert
**Estimated time:** 3 weeks

## Goal
Take models and AI apps from a notebook to a reliable, observable, cost-controlled production system.

## Topics
### Serving
- **FastAPI** for ML APIs (async, Pydantic, OpenAPI)
- **BentoML**, **Ray Serve**, **Triton Inference Server**
- LLM-specific: **vLLM**, **TGI**, **TensorRT-LLM**
- Batching, continuous batching, paged attention
- Streaming responses (SSE, websockets)

### Containerization
- Docker: multi-stage builds, slim base images, GPU images
- Docker Compose for local stacks
- Image scanning, vulnerability management

### Orchestration
- Kubernetes basics: pods, deployments, services, HPA
- Helm charts
- KServe / Seldon for ML on Kubernetes

### CI/CD for ML
- **GitHub Actions** workflows for tests + linting + deploy
- Model registries: MLflow, Hugging Face Hub
- Promoting models through environments (dev → staging → prod)
- Canary & shadow deployments

### Data & feature pipelines
- Orchestrators: **Airflow**, **Prefect**, **Dagster**
- Feature stores: **Feast**, **Tecton** (when you actually need one)
- Streaming: Kafka basics

### Monitoring & observability
- Logs, metrics, traces (the three pillars)
- **Prometheus** + **Grafana** for infra metrics
- **OpenTelemetry** for traces
- LLM observability: **Langfuse**, **LangSmith**, **Helicone**, **Arize**
- Drift detection: data drift, prediction drift, concept drift
- Quality monitoring: hallucination rate, eval scores in prod
- Cost & latency dashboards

### Reliability
- Retries, timeouts, circuit breakers
- Rate limiting, quotas
- Fallbacks (smaller model, cached response)
- Load testing (`locust`, `k6`)

## Exercises
1. Wrap a model in FastAPI with proper Pydantic schemas; deploy in Docker.
2. Add Prometheus metrics + a Grafana dashboard for QPS, p95 latency, error rate.
3. Set up GitHub Actions to run tests + an eval suite on every PR.
4. Add Langfuse tracing to a RAG or agent app from earlier modules.
5. Load-test the API and tune batch size for throughput vs latency.

## Resources
- FastAPI docs: https://fastapi.tiangolo.com/
- vLLM docs: https://docs.vllm.ai/
- Book: *Designing Machine Learning Systems* — Chip Huyen
- Book: *Machine Learning Engineering* — Andriy Burkov

## Checkpoint
You can deploy a model or LLM app behind a versioned API with CI/CD, metrics, traces, and a clear rollback path.
