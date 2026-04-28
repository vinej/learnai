"""
06 — Scaling patterns

When traffic grows, the failure modes change. A list of patterns and when
each helps.

Run: python 06_scaling_patterns.py
"""

PATTERNS = """\
=== HORIZONTAL SCALING ===

The default. Run more replicas of a stateless API behind a load balancer.

  - Each replica is identical, holds no per-user state.
  - Use a shared cache (Redis, Memcached) and a shared DB (Postgres) for
    state.
  - Auto-scale replicas based on CPU, queue depth, or RPS.

  Tooling: Kubernetes HPA (CPU / custom metrics), AWS Application Auto
  Scaling, Cloud Run revisions.


=== AUTOSCALING ===

Two flavors:
  - REACTIVE: scale up when CPU > 70% for N minutes. Lags spikes.
  - PREDICTIVE: scale up by schedule or by ML forecast. Avoid cold starts
    during predictable surges (e.g. 9am Monday).

Cold-start mitigations:
  - Keep a MIN replica count > 0.
  - Pre-warm: send synthetic traffic during scale-up.
  - For LLMs: keep weights mmap'd; serverless cold starts are punishing
    on multi-GB models.


=== MULTI-REGION SERVING ===

Latency-sensitive traffic should hit a region close to the user.

  - DNS-based routing (Route53 latency / GeoDNS) for the simple case.
  - Anycast IPs (Cloudflare, Fastly) for global edge.
  - Replicate model weights and data PER REGION. Bandwidth is cheaper than
    cross-region request latency.

  Trade-offs:
  - Data residency: GDPR / HIPAA may pin data to specific regions.
  - Cost: full replication of ML weights × N regions adds up.


=== GPU SHARING / FRACTIONAL GPUs ===

A single H100 has 80GB of HBM. A 7B model takes ~14GB in bf16. Don't waste 66GB.

Options:
  - Multi-Process Service (MPS) — multiple processes share one GPU.
  - MIG (Multi-Instance GPU) — H100 / A100 hardware partitioning.
  - vLLM continuous batching — different requests share KV cache pages.
  - Fractional schedulers (Run:AI, Bcube) — orchestrate the above on k8s.

Rule of thumb:
  - For self-hosted LLM serving, packing requests via vLLM is the biggest win.
  - For training, fractional GPUs help dev / experimentation; production
    training takes whole GPUs (or many).


=== SPOT vs ON-DEMAND ===

  Spot instances are 60-90% cheaper but can be reclaimed at any moment.

  Use spot for:
    - Batch / training jobs that checkpoint often.
    - Stateless inference replicas (with graceful drain).
    - Embedding generation at scale.

  Don't use spot for:
    - The PRIMARY copy of a real-time API.
    - Workloads with strict SLAs unless you have a shadow on-demand fleet.

  Pattern:
    - Mix: 70% spot for cost, 30% on-demand for resilience. AWS, GCP, Azure
      all have managed spot pools.


=== CONNECTION POOLING + PIPELINING ===

For servers calling LLM APIs:
  - Reuse HTTPS connections (httpx.AsyncClient + keep-alive).
  - Use the SDK's `AsyncAnthropic` to pipeline many concurrent requests
    through a small number of TCP connections.
  - Tune the connection pool size: too small → head-of-line blocking,
    too big → file-descriptor exhaustion.


=== QUEUE-BASED LEVELING ===

When clients can wait, push requests onto a QUEUE (SQS, Redis Streams,
Kafka, RabbitMQ). Workers drain it.

  - Smooths spikes — the queue absorbs bursts.
  - Easy to scale workers independently.
  - Easy retries: failed messages go to a dead-letter queue.

  Use for: batch summarization, async report generation, embeddings backfill.
  Avoid for: real-time chat (latency).


=== READ-MOSTLY: CACHING TIERS ===

Three levels:
  L1: in-process LRU cache (ms latency, single-replica).
  L2: shared cache (Redis / Memcached) — multi-replica.
  L3: object store (S3, GCS) — durable, cheap, slower.

Use them for:
  - Tokenized prompts (rare; usually cached server-side).
  - Embedding vectors for known docs.
  - Pre-computed answers for FAQ-style queries.


=== INVENTORY: THE "USE WHEN" CHECKLIST ===

Symptom: latency rises with QPS.
  → Add replicas (horizontal). Profile for hot spots.

Symptom: replicas are at 30% CPU but lots of pending requests.
  → Async I/O issue. Make the LLM call non-blocking.

Symptom: cost dominates revenue.
  → Cascade + caching + smaller models (file 05).

Symptom: spikes cause cold-start storms.
  → Min replicas > 0; pre-warm.

Symptom: cross-region users hit slow latency.
  → Regional replicas + GeoDNS.

Symptom: training cluster sits idle 80% of the time.
  → Spot pool; queue smaller jobs through it.
"""

print(PATTERNS)


# ───────────────────────────────────────────────────────────
# A reference architecture diagram (in text)
# ───────────────────────────────────────────────────────────
ARCH = """\
                                 ┌──────────────┐
       Client  ───────►   CDN  ─►│ Edge proxy   │── auth, rate-limit
                                 │ (Cloudflare) │
                                 └──────┬───────┘
                                        │
                                  Region selector (GeoDNS / latency)
                                        │
                ┌─────────────────┬─────┼─────┬───────────────┐
                ▼                 ▼           ▼               ▼
          us-east-1          us-west-2    eu-west-1     ap-southeast-1
          ┌──────┐            ┌──────┐    ┌──────┐      ┌──────┐
          │ K8s  │            │ K8s  │    │ K8s  │      │ K8s  │
          │ API  │            │ API  │    │ API  │      │ API  │
          └──┬───┘            └──┬───┘    └──┬───┘      └──┬───┘
             │                   │           │             │
        ┌────┴──────────┐        │           │             │
        ▼               ▼        ▼           ▼             ▼
   ┌────────┐     ┌──────────┐
   │ Redis  │     │ Postgres │   (per-region replicas of state stores)
   │ cache  │     │  + pgvec │
   └────────┘     └──────────┘
                                  │
                                  ▼
                           ┌────────────────┐
                           │ Anthropic API  │ (managed, multi-region)
                           └────────────────┘
                                  │
                                  ▼
                           ┌────────────────┐
                           │ Self-hosted    │   (vLLM / TGI fleet,
                           │ models         │    if cost demands it)
                           └────────────────┘

Cross-cutting:
  • Observability:  Prometheus + Grafana + Loki + Langfuse.
  • Secrets:        Vault / AWS SM. Never in code or images.
  • CI/CD:          GitHub Actions → ghcr.io → ArgoCD per region.
"""
print(ARCH)
