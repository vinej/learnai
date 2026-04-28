"""
Exercise 5 — Async load-test the FastAPI predictor

Driver that hammers Exercise 1's endpoint at a configurable concurrency.
Reports:
  • throughput (RPS)
  • latency P50 / P95 / P99
  • error rate

Asserts on a smoke run that the service handled ≥ 95% of requests
successfully under modest concurrency.

Steps:
  # In one terminal:
  python exercises/01_fastapi_predict.py train
  python exercises/01_fastapi_predict.py serve

  # In another:
  python exercises/05_load_test.py
"""

import asyncio
import statistics
import sys
import time

import httpx


URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"bedrooms": 3, "bathrooms": 2, "sqft": 1800, "lot_size": 7000}


async def hammer(client: httpx.AsyncClient, n_requests: int) -> list[tuple[bool, float]]:
    results = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json=PAYLOAD, timeout=10)
            ok = r.status_code == 200
        except Exception:
            ok = False
        results.append((ok, (time.perf_counter() - t0) * 1000))
    return results


async def run(concurrency: int, total: int):
    per_worker = total // concurrency
    async with httpx.AsyncClient(http2=False) as client:
        # Warm up
        await client.post(URL, json=PAYLOAD, timeout=5)

        t0 = time.perf_counter()
        all_results: list[tuple[bool, float]] = []
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(hammer(client, per_worker))
                     for _ in range(concurrency)]
        for t in tasks:
            all_results.extend(t.result())
        elapsed = time.perf_counter() - t0
    return all_results, elapsed


def report(results, elapsed):
    n = len(results)
    n_ok = sum(1 for ok, _ in results if ok)
    rate_ok = n_ok / n
    latencies_ms = sorted([lat for _, lat in results if lat > 0])

    def percentile(p):
        if not latencies_ms:
            return 0.0
        i = max(0, int(len(latencies_ms) * p / 100) - 1)
        return latencies_ms[i]

    print(f"\n=== results ===")
    print(f"  total:       {n} requests in {elapsed:.2f}s")
    print(f"  throughput:  {n / elapsed:.1f} RPS")
    print(f"  success:     {n_ok}/{n}  ({rate_ok:.1%})")
    print(f"  errors:      {n - n_ok}")
    print(f"  P50 latency: {percentile(50):.1f} ms")
    print(f"  P95 latency: {percentile(95):.1f} ms")
    print(f"  P99 latency: {percentile(99):.1f} ms")
    print(f"  mean:        {statistics.mean(latencies_ms):.1f} ms" if latencies_ms else "")
    return rate_ok


async def main():
    print(f"loading {URL}: 200 reqs over 10 concurrent clients")
    try:
        results, elapsed = await run(concurrency=10, total=200)
    except httpx.ConnectError:
        print("ERROR: connect refused. Is the server running?")
        print("  python exercises/01_fastapi_predict.py serve")
        sys.exit(1)

    rate_ok = report(results, elapsed)

    assert rate_ok >= 0.95, f"success rate {rate_ok:.1%} below 95%"
    print("\nload test passed ✅")


if __name__ == "__main__":
    asyncio.run(main())


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Try concurrency 1, 10, 50, 200. Plot throughput vs latency. Find the knee.
# • Add a `--rps` flag and rate-limit the requests with aiolimiter.
# • Switch to locust for a richer scenario:
#     pip install locust
#     locustfile.py:
#         from locust import HttpUser, task
#         class User(HttpUser):
#             @task
#             def predict(self):
#                 self.client.post("/predict", json={...})
#     locust -H http://localhost:8000 --users 50 --spawn-rate 10
# • Run the load test from inside Docker — closer to production conditions.
# ─────────────────────────────────────────────────────────────────────────────
