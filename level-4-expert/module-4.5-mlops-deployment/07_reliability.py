"""
07 — Reliability patterns

LLM and ML APIs WILL fail. Networks blink. Providers throttle. Models OOM.
A reliable stack assumes failure and stays mostly-up.

Five patterns you'll need:

  RETRIES with EXPONENTIAL BACKOFF + JITTER  — for transient failures.
  TIMEOUTS                                    — never wait forever.
  CIRCUIT BREAKERS                            — stop hammering a dead service.
  RATE LIMITING / QUOTAS                       — protect your infra and costs.
  FALLBACKS                                    — degrade gracefully.

This file shows each, in plain Python, then wire-ups for production.

Run: python 07_reliability.py
"""

import asyncio
import random
import time

import httpx


# ───────────────────────────────────────────────────────────
# 1. Retries with backoff + jitter
# ───────────────────────────────────────────────────────────
class Retryable(Exception):
    pass


def with_retries(max_attempts: int = 4, base_delay: float = 0.2):
    def deco(fn):
        def inner(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Retryable as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    delay += random.uniform(0, base_delay)   # jitter
                    print(f"  retry attempt {attempt + 1}/{max_attempts} after {delay:.2f}s ({e})")
                    time.sleep(delay)
        return inner
    return deco


@with_retries(max_attempts=4)
def flaky_call():
    if random.random() < 0.6:
        raise Retryable("transient")
    return "ok"


print("=== retries with exponential backoff + jitter ===")
print(flaky_call())


# ───────────────────────────────────────────────────────────
# 2. Timeouts — every external call has one
# ───────────────────────────────────────────────────────────
TIMEOUT_NOTES = """\
- httpx / aiohttp: pass `timeout=10` (seconds).
- anthropic SDK: `Anthropic(timeout=30.0)` overrides default 600s.
- DB drivers: Postgres `statement_timeout`; Redis `socket_timeout`.
- Outermost request handler: a HARD timeout (e.g. 60s) drops the request,
  releases the worker, surfaces a clean error to the caller.
"""
print(TIMEOUT_NOTES)


# ───────────────────────────────────────────────────────────
# 3. Circuit breaker
# ───────────────────────────────────────────────────────────
# Track recent failures. If too many fail, OPEN the breaker → reject calls
# instantly for a cooldown. After cooldown, allow ONE attempt; if it
# succeeds, CLOSE again.
class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = range(3)

    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at = 0.0
        self.state = self.CLOSED

    def call(self, fn, *args, **kwargs):
        if self.state == self.OPEN:
            if time.time() - self.opened_at < self.cooldown:
                raise RuntimeError("circuit_open")
            self.state = self.HALF_OPEN

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if self.state == self.HALF_OPEN or self.failures >= self.threshold:
                self.state = self.OPEN
                self.opened_at = time.time()
                self.failures = 0
            raise

        # Success
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
        self.failures = 0
        return result


print("\n=== circuit breaker ===")
breaker = CircuitBreaker(threshold=2, cooldown=1)


def always_fails():
    raise RuntimeError("dead")


for i in range(5):
    try:
        breaker.call(always_fails)
    except Exception as e:
        print(f"  attempt {i + 1}: {e}")


# ───────────────────────────────────────────────────────────
# 4. Rate limiting — token bucket
# ───────────────────────────────────────────────────────────
class TokenBucket:
    def __init__(self, capacity: int, refill_per_s: float):
        self.capacity = capacity
        self.refill_rate = refill_per_s
        self.tokens = capacity
        self.last = time.time()

    def take(self, n: int = 1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


print("\n=== token bucket rate limit ===")
bucket = TokenBucket(capacity=3, refill_per_s=1)
allowed = sum(1 for _ in range(10) if bucket.take())
print(f"  10 attempts, {allowed} allowed (capacity 3, refill 1/s)")


# ───────────────────────────────────────────────────────────
# 5. Fallbacks — graceful degradation
# ───────────────────────────────────────────────────────────
FALLBACKS = """\
THE LADDER (LLM example):

  1. Primary: Claude Sonnet 4.6 (best quality).
  2. If rate-limited or 5xx → Claude Haiku 4.5 (cheaper, almost as good).
  3. If still failing → return a CACHED similar response (lower-quality but
     still useful).
  4. If even that fails → return a "we're degraded; try again" message
     with a status link.

Each tier should have its own timeout and circuit breaker. Don't cascade
through 4 retries × 4 tiers × 30s — your latency budget is gone.
"""
print(FALLBACKS)


# ───────────────────────────────────────────────────────────
# Production-grade libraries
# ───────────────────────────────────────────────────────────
LIBS = """\
- tenacity         — retries with backoff (Python). Drop-in decorator.
- httpx             — modern async HTTP with timeouts, retries.
- pybreaker         — circuit breakers.
- slowapi           — FastAPI rate limiting.
- aiolimiter         — async token-bucket limiter.

API-side:
- nginx, Cloudflare, Envoy   — edge rate limiting.
- AWS WAF, CloudFront         — managed.
- Redis-based limiters         — distributed rate limit across replicas.
"""
print(LIBS)


# ───────────────────────────────────────────────────────────
# Load-test BEFORE shipping
# ───────────────────────────────────────────────────────────
LT = """\
Tools:
  locust   — Python, scriptable scenarios, web UI.
  k6        — JS / TS, single binary, great CI integration.
  vegeta    — CLI; one-liner to spike a URL.
  artillery — Node.js, similar to locust.

What to measure:
  • P50 / P95 / P99 latency at increasing RPS.
  • Where does the CPU bottleneck — your code, the LLM, the DB?
  • Saturation: queue depth, dropped requests.
  • Error budget: 1% errors at peak might be fine for an internal tool;
    not for a payments API.

Test from a place that mirrors production: same region, same network,
behind your real proxy. Localhost numbers lie.
"""
print(LT)
