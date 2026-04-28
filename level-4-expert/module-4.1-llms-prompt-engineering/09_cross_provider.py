"""
09 — Cross-provider patterns

The big providers (Anthropic, OpenAI, Google, AWS Bedrock, Azure OpenAI,
xAI, Mistral) and open-source servers (vLLM, Ollama, llama.cpp) all expose
broadly similar abstractions:

  - chat-style messages (system, user, assistant).
  - tool / function calling.
  - streaming.
  - structured outputs.

The DETAILS differ — request fields, model names, content-block shapes,
caching syntax, billing.

Two ways to handle multi-provider code:

  1. Use each native SDK directly. Best for advanced features (caching,
     extended thinking, region-specific endpoints).
  2. Use a shim like LiteLLM, OpenRouter, or LangChain's chat models. One
     API for many providers — at the cost of feature lag.

This file walks the patterns; no API call needed.

Run: python 09_cross_provider.py
"""

# ───────────────────────────────────────────────────────────
# Anthropic — what you've used in this module
# ───────────────────────────────────────────────────────────
ANTHROPIC = """\
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    system="You are concise.",
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.content[0].text)
"""

# ───────────────────────────────────────────────────────────
# OpenAI
# ───────────────────────────────────────────────────────────
OPENAI = """\
from openai import OpenAI
client = OpenAI()        # reads OPENAI_API_KEY
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=200,
    messages=[
        {"role": "system",  "content": "You are concise."},
        {"role": "user",    "content": "Hi"},
    ],
)
print(resp.choices[0].message.content)
"""

# ───────────────────────────────────────────────────────────
# Google Gemini (via google-genai)
# ───────────────────────────────────────────────────────────
GEMINI = """\
from google import genai
client = genai.Client()  # reads GOOGLE_API_KEY
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Hi"],
)
print(resp.text)
"""


# ───────────────────────────────────────────────────────────
# Self-hosted models (Ollama / vLLM / llama.cpp)
# ───────────────────────────────────────────────────────────
# All three expose an OpenAI-compatible HTTP endpoint, so the OpenAI SDK
# works against them with a different base_url.
OPENAI_COMPATIBLE = """\
from openai import OpenAI
# Ollama runs locally on :11434
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="llama3.1",          # whichever model you've pulled
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
"""


# ───────────────────────────────────────────────────────────
# LiteLLM — one API for many providers
# ───────────────────────────────────────────────────────────
# pip install litellm
LITELLM = """\
import litellm
# auto-routes to whichever provider matches the model prefix
for model in ("anthropic/claude-haiku-4-5-20251001",
              "openai/gpt-4o-mini",
              "gemini/gemini-2.5-flash"):
    resp = litellm.completion(model=model,
                              messages=[{"role": "user", "content": "Hi"}])
    print(model, "→", resp.choices[0].message.content[:80])
"""


print("=== same chat call, four flavors ===")
for label, code in [
    ("Anthropic SDK",       ANTHROPIC),
    ("OpenAI SDK",          OPENAI),
    ("Google Gemini SDK",   GEMINI),
    ("OpenAI-compatible (Ollama / vLLM / llama.cpp)", OPENAI_COMPATIBLE),
    ("LiteLLM (one API for all)", LITELLM),
]:
    print(f"\n--- {label} ---\n{code}")


# ───────────────────────────────────────────────────────────
# What differs across providers (you'll have to handle it)
# ───────────────────────────────────────────────────────────
# • SYSTEM PROMPT shape: top-level field (Anthropic) vs first message (OpenAI).
# • TOOL CALLING shape: tool_use / tool_result blocks (Anthropic) vs
#   function_call (OpenAI deprecated) vs tool_calls / tool message role (OpenAI new).
# • PROMPT CACHING: Anthropic, OpenAI, and Gemini all support it but with
#   different syntax and lifetimes.
# • EXTENDED THINKING: Anthropic-specific.
# • IMAGES: very similar (base64 / URL); some providers also accept files.
# • RATE LIMITS, regional endpoints, billing models — all differ.


# ───────────────────────────────────────────────────────────
# When to use SDKs directly vs an abstraction
# ───────────────────────────────────────────────────────────
# DIRECT SDK
#   You're building serious features that depend on provider-specific
#   capabilities (caching, extended thinking, batch, embeddings, files API).
#   You want exact billing visibility and the latest features the day they
#   ship.
#
# ABSTRACTION (LiteLLM, LangChain, OpenRouter)
#   You're building a product that needs to RUN ANY MODEL the user picks.
#   You want one chunk of code, a config file changes the brain.
#   You're prototyping; ergonomics matter more than 100% feature parity.
#
# Many production apps use BOTH: their fast path uses the native SDK for
# Anthropic Claude, and a fallback path uses LiteLLM to route to a backup
# provider when the primary is down.


# ───────────────────────────────────────────────────────────
# A friendly default architecture
# ───────────────────────────────────────────────────────────
# Build a small `LLM` interface in your codebase:
#
#   class LLM(Protocol):
#       def complete(self, system: str, messages: list[Message]) -> Response: ...
#
# Implement it once per provider you actually use. Production code calls
# `llm.complete(...)`, never the SDK directly. Swapping providers means
# writing one new implementation, not rewriting your code.
