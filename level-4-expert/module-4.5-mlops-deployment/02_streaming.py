"""
02 — Streaming responses

For LLMs, you almost always want STREAMING — the user sees the first
token in 200ms instead of waiting 5+ seconds for the full reply.

FastAPI exposes two patterns:

  StreamingResponse + a generator       — easy, plain text.
  EventSourceResponse (SSE, sse-starlette) — Server-Sent Events with named
                                              event types. Best for LLM UIs.
  WebSocket                              — bidirectional. Use for chat where
                                              the client sends interruptions.

This file shows the SSE pattern that 90% of LLM apps use.

Run: python 02_streaming.py
"""

EXAMPLE = """\
# stream_server.py
from typing import AsyncIterator

import anthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


app = FastAPI()
client = anthropic.AsyncAnthropic()


class ChatRequest(BaseModel):
    message: str


async def stream_claude(message: str) -> AsyncIterator[str]:
    \"\"\"Yield each text delta as a Server-Sent Event.\"\"\"
    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        async for text in stream.text_stream:
            # SSE format: each event is `data: <payload>\\n\\n`
            yield f"data: {text}\\n\\n"
        # Sentinel so the client knows we're done
        yield "data: [DONE]\\n\\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_claude(req.message),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},   # disable nginx buffering
    )
"""

print("=== streaming Claude through FastAPI as SSE ===")
print(EXAMPLE)


# ───────────────────────────────────────────────────────────
# Client side
# ───────────────────────────────────────────────────────────
CLIENT_CODE = """\
# Browser JS:
const resp = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: "Hello"}),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // text contains "data: <chunk>\\n\\n" lines
    process(text);
}

# Python client (httpx async):
import httpx
async with httpx.AsyncClient() as client:
    async with client.stream("POST", "http://.../chat",
                             json={"message": "Hello"}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                print(payload, end="", flush=True)
"""
print("\n=== client examples ===")
print(CLIENT_CODE)


# ───────────────────────────────────────────────────────────
# Why SSE over WebSockets?
# ───────────────────────────────────────────────────────────
COMPARE = """\
SSE (Server-Sent Events) — one-way, server → client.
  + Plain HTTP. Works through proxies, load balancers, CDNs.
  + Auto-reconnect built into the EventSource browser API.
  + Easy to log and replay.
  - One-way only. No client interruption mid-response without a separate POST.

WebSocket — bidirectional, persistent connection.
  + Client can send interruptions / new turns mid-stream.
  + Lower per-message overhead.
  - Some proxies don't pass it cleanly. CDN compatibility uneven.
  - Stickiness with a load balancer required.

For a one-way LLM stream where the user just watches the response unfold:
  SSE.

For real-time multi-turn (live transcription, voice agents, code execution
agents that the user can interrupt):
  WebSocket.
"""
print(COMPARE)


# ───────────────────────────────────────────────────────────
# Operational concerns
# ───────────────────────────────────────────────────────────
OPS = """\
TIMEOUTS
  Set a CLIENT timeout: 120s for chat, 600s for long agent runs.
  Set a SERVER timeout: longer than the model's max_tokens / decode rate.

DISCONNECTS
  Detect cancelled requests: in FastAPI, `await request.is_disconnected()`
  inside the generator. STOP THE LLM CALL when the client leaves —
  you're paying for tokens.

BUFFERING
  X-Accel-Buffering: no   (nginx)
  Cache-Control: no-store  (CDNs)
  Some proxies still buffer; test through your real path.

MIDDLEWARE
  Don't put response-buffering middleware in front of streaming endpoints
  (e.g., gzip via FastAPI's GZipMiddleware will buffer the whole response).

OBSERVABILITY
  Log: model, prompt length, time-to-first-token, total time, total tokens,
  finish_reason, client cancellation.
"""
print(OPS)


# ───────────────────────────────────────────────────────────
# Putting it all together
# ───────────────────────────────────────────────────────────
# A production LLM endpoint usually has:
#   • An /chat POST that opens an SSE stream.
#   • Authentication (API key or OAuth).
#   • Rate limiting per user/key.
#   • Cancellation handling on client disconnect.
#   • Per-request trace ID propagated to logs.
#   • Metrics: time_to_first_token, tokens_per_second, output_tokens.
