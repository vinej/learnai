"""
03 — First calls with the Anthropic Python SDK

The basic pattern:

  client = anthropic.Anthropic()                       # reads ANTHROPIC_API_KEY
  response = client.messages.create(
      model="claude-haiku-4-5-20251001",
      max_tokens=512,
      messages=[{"role": "user", "content": "Hello!"}],
  )
  print(response.content[0].text)

This file:
  • Makes a basic call.
  • Reads the response (text + usage + stop_reason).
  • Demonstrates STREAMING — start showing tokens as they're produced.

Run: python 03_anthropic_basics.py    # requires ANTHROPIC_API_KEY
"""

import anthropic

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# 1. The most basic call
# ───────────────────────────────────────────────────────────
print("=== basic call ===")
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=256,
    messages=[
        {"role": "user", "content": "In one sentence, what's the capital of France?"},
    ],
)
print("response:", response.content[0].text)
print(f"stop_reason: {response.stop_reason}")
print_usage("basic", response.usage)


# ───────────────────────────────────────────────────────────
# 2. With a system prompt
# ───────────────────────────────────────────────────────────
print("\n=== with system prompt ===")
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=256,
    system="You speak only in haiku — three lines, 5/7/5 syllables.",
    messages=[
        {"role": "user", "content": "Tell me about Paris."},
    ],
)
print(response.content[0].text)
print_usage("haiku", response.usage)


# ───────────────────────────────────────────────────────────
# 3. Multi-turn conversation
# ───────────────────────────────────────────────────────────
# Keep your own list of {"role": ..., "content": ...} messages and pass the
# whole list each turn. Always alternate user / assistant after the first user
# message.
print("\n=== multi-turn ===")
messages = [
    {"role": "user", "content": "I'm planning a 3-day trip to Tokyo. What's the first thing I should know?"},
]
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=messages,
)
assistant_reply = response.content[0].text
print("assistant:", assistant_reply.strip())

messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "Now suggest one neighborhood I shouldn't miss."})

response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=messages,
)
print("\nassistant:", response.content[0].text.strip())
print_usage("turn-2", response.usage)


# ───────────────────────────────────────────────────────────
# 4. Streaming — the right default for any user-facing UI
# ───────────────────────────────────────────────────────────
# `messages.stream` returns a context manager that yields incremental events.
# `stream.text_stream` is a convenience iterator over just the text deltas.
print("\n=== streaming ===")
with client.messages.stream(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Count slowly from one to seven, one number per line."},
    ],
) as stream:
    print("(streaming) ", end="", flush=True)
    for chunk in stream.text_stream:
        print(chunk, end="", flush=True)
    print()
    final = stream.get_final_message()
print_usage("stream", final.usage)


# ───────────────────────────────────────────────────────────
# 5. Stop reasons — handle them
# ───────────────────────────────────────────────────────────
# stop_reason can be:
#   "end_turn"      — model decided it was done.
#   "max_tokens"    — output ran out before the model finished.
#   "stop_sequence" — hit a sequence you specified.
#   "tool_use"      — model wants to use a tool (file 07).
#
# If you see max_tokens often, raise the limit OR ask the model to be brief.


# ───────────────────────────────────────────────────────────
# 6. Errors
# ───────────────────────────────────────────────────────────
# Wrap calls in try/except to handle:
#   anthropic.APIConnectionError    — network issue
#   anthropic.RateLimitError        — back off / retry with jitter
#   anthropic.APIStatusError        — 4xx/5xx; surface to user
# The SDK retries 5xx with backoff by default. You can tune via
# `Anthropic(max_retries=N, timeout=T)`.
