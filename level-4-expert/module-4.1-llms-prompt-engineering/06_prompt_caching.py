"""
06 — Prompt caching

If you reuse a long prompt across many calls (system instructions, document
context, few-shot examples), prompt caching can save up to 90% on input
cost and significantly reduce latency.

How it works:
  • You mark a content block with `cache_control: {"type": "ephemeral"}`.
  • The first call CREATES a cache entry (writes are slightly more expensive
    than normal input).
  • Subsequent calls within the cache lifetime READ the cache (much cheaper
    than normal input).

Default lifetime: 5 minutes. Optionally `"ttl": "1h"` for hour-long cache.
You can place up to 4 cache breakpoints in a single request.

Order matters: anything BEFORE a cache breakpoint is cached together.
Tools → system → messages — each can be a cache point.

Run: python 06_prompt_caching.py    # requires ANTHROPIC_API_KEY
"""

import time

import anthropic

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# A "long" system prompt — fictional company knowledge
# ───────────────────────────────────────────────────────────
# Caches need a minimum number of tokens (1024 for Sonnet/Opus, 2048 for
# Haiku). Below the threshold, cache_control is silently ignored. We pad
# the prompt to make sure we hit the bar for Haiku.
LONG_SYSTEM = (
    "You are a customer-support assistant for Acme Corp.\n\n"
    "Acme Corp is a fictional B2B SaaS company that sells a developer "
    "platform called Cogwheel. Cogwheel has three plans (Starter, Team, "
    "Enterprise) priced at $29, $99, and 'contact us'. The product "
    "supports SAML SSO on Team and Enterprise plans only. The free trial "
    "is 14 days, no credit card required. "
    + "Customer-support standard answers about every imaginable feature. "
      "Refund policy: pro-rata refunds within 30 days; no refunds after. "
      "API rate limits: 60 RPM on Starter, 600 on Team, custom on Enterprise. "
      "Support hours: 24/7 chat; phone Mon-Fri 9am-6pm PT for Enterprise. "
      "Backups: daily incremental, 30-day retention; PITR for Enterprise. "
      "Compliance: SOC 2 Type II, GDPR, HIPAA-eligible on Enterprise. "
      "Available regions: us-east, us-west, eu-west, ap-southeast. "
      "Single sign-on: Okta, Azure AD, Google Workspace, generic SAML 2.0. "
      "Two-factor auth: TOTP and WebAuthn supported on all plans. " * 30
)
print(f"long system prompt: ~{len(LONG_SYSTEM)} chars (~{len(LONG_SYSTEM) // 4} tokens)")


# ───────────────────────────────────────────────────────────
# Helper: ask a question with caching enabled
# ───────────────────────────────────────────────────────────
def ask(question: str, label: str):
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=200,
        # IMPORTANT: when system is a list of blocks, you can attach cache_control
        # to ANY block. Mark the long block as cacheable.
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": question}],
    )
    print_usage(label, response.usage)
    return response


# ───────────────────────────────────────────────────────────
# Run two calls with the same long system prompt
# ───────────────────────────────────────────────────────────
print("=== call 1: cache miss (writes the cache) ===")
ask("What's the refund policy?", "first")

# A small delay to make the cache state obvious
time.sleep(1)

print("\n=== call 2: cache hit (reads the cache, much cheaper) ===")
ask("Which auth methods are supported?", "second")


# ───────────────────────────────────────────────────────────
# Verifying a cache hit
# ───────────────────────────────────────────────────────────
# response.usage gives:
#   input_tokens                  — fresh tokens this turn (uncached)
#   cache_creation_input_tokens   — tokens just written to the cache
#   cache_read_input_tokens       — tokens served from the cache
#
# A successful cache hit = cache_read_input_tokens == big number,
# input_tokens == only the new user message.


# ───────────────────────────────────────────────────────────
# Where to place cache breakpoints
# ───────────────────────────────────────────────────────────
# In order from most-cacheable to least:
#   1. tools (function definitions)
#   2. system prompt
#   3. early messages (e.g. retrieved docs)
#   4. recent user/assistant turns
#
# Cache from MOST STABLE to LEAST. Anything AFTER a breakpoint
# automatically invalidates the cache when it changes.
#
# Multiple breakpoints (up to 4) help when DIFFERENT parts of the prompt
# change at different rates: tools change rarely, the system prompt rarely,
# the conversation turns frequently.


# ───────────────────────────────────────────────────────────
# Long-form caching (1-hour TTL)
# ───────────────────────────────────────────────────────────
# For prompts that survive longer than 5 min:
#   "cache_control": {"type": "ephemeral", "ttl": "1h"}
# Costs slightly more on cache CREATION; same discount on reads.
# Useful for batch jobs, periodic summaries, expensive document corpora.


# ───────────────────────────────────────────────────────────
# What does NOT cache well
# ───────────────────────────────────────────────────────────
# • Anything below the per-model minimum (Haiku 2048, Sonnet/Opus 1024 tokens).
# • Prompts whose CONTENT changes per call (timestamps, user IDs interleaved).
# • Anything AFTER a recent message — only stable PREFIXES cache.
#
# Restructure prompts so the dynamic part is at the END.


# ───────────────────────────────────────────────────────────
# Real-world impact
# ───────────────────────────────────────────────────────────
# RAG: cache the retrieved-doc context (5-50k tokens) per session.
#       Sub-second responses, ~90% input cost cut.
# Support bots: cache the policy doc + persona. Fresh user query at the end.
# Code-review agents: cache the repo structure / diff. Vary the question.
# Multi-step agents: cache tool definitions and few-shot examples.
