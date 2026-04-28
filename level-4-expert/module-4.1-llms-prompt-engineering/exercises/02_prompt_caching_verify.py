"""
Exercise 2 — Verify prompt caching saves money

Make a long system prompt, send 3 different user questions in a row, and
verify:

  call 1 — cache_creation_input_tokens > 0  (writing the cache)
  call 2 — cache_read_input_tokens > 0     (reading the cache)
  call 3 — cache_read_input_tokens > 0     (still reading the cache)

The script asserts the SECOND and THIRD calls have non-zero
cache_read_input_tokens — i.e. caching actually engaged.

Run: python exercises/02_prompt_caching_verify.py    # requires ANTHROPIC_API_KEY
"""

import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


# ───────────────────────────────────────────────────────────
# A long, stable, cacheable system prompt
# ───────────────────────────────────────────────────────────
# The Haiku cache minimum is 2048 tokens. We pad with realistic-looking
# fake policy text to clear that bar.
LONG_SYSTEM = (
    "You are a senior engineer at FluffCorp, a fictional company. "
    "Use ONLY the policies below; never invent facts.\n\n"
    "## Policies\n"
    + (
        "* All deployments must be peer-reviewed.\n"
        "* No direct pushes to main on weekends.\n"
        "* Backups run nightly at 02:00 UTC and retain 30 days.\n"
        "* Database migrations require ops sign-off.\n"
        "* Secrets are stored in Vault — never in env files.\n"
        "* Rate limits are 60 RPM per IP on the public API.\n"
        "* The CI pipeline runs lint, types, tests, and security checks.\n"
        "* Pull requests need at least one approving review.\n"
        "* Major releases announce in #engineering 24 hours in advance.\n"
        "* Production incidents follow the SEV1-SEV5 ladder.\n"
        * 30
    )
)


def ask_with_cache(client, question: str, label: str):
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    u = response.usage
    print(
        f"  [{label}] input={u.input_tokens}  output={u.output_tokens}  "
        f"cache_create={u.cache_creation_input_tokens}  "
        f"cache_read={u.cache_read_input_tokens}  "
        f"≈ ${cost_of_usage(u):.5f}"
    )
    return u


def main():
    require_api_key()
    client = anthropic.Anthropic()

    questions = [
        "What's the deployment review policy?",
        "Where are secrets stored?",
        "When do backups run?",
    ]

    print(f"system prompt: ~{len(LONG_SYSTEM) // 4} tokens (target ≥ 2048 for Haiku)\n")
    print("running 3 calls with the same cacheable system prompt:")

    usages = []
    for i, q in enumerate(questions, start=1):
        usages.append(ask_with_cache(client, q, f"call {i}"))
        time.sleep(0.5)

    # ── Assertions ──
    # First call should write the cache.
    assert usages[0].cache_creation_input_tokens and usages[0].cache_creation_input_tokens > 0, (
        "first call should have CREATED a cache entry. "
        "If this fails, your system prompt may be below the minimum cacheable size."
    )
    # Second and third calls should read it.
    for i in (1, 2):
        assert usages[i].cache_read_input_tokens and usages[i].cache_read_input_tokens > 0, (
            f"call {i+1} should have READ from the cache"
        )
        # Fresh input tokens for these calls should be SMALL (just the user msg).
        assert (usages[i].input_tokens or 0) < 200, (
            f"call {i+1}'s fresh input_tokens={usages[i].input_tokens} — "
            "expected the long system prompt to come from cache, not fresh."
        )

    cached_tokens_total = sum(u.cache_read_input_tokens or 0 for u in usages[1:])
    fresh_input_total   = sum(u.input_tokens or 0 for u in usages)
    print(f"\ntotal fresh input tokens across 3 calls: {fresh_input_total}")
    print(f"total cache-READ input tokens:           {cached_tokens_total}")
    print("→ caching saved input billing on calls 2 and 3 ✅")


if __name__ == "__main__":
    main()
