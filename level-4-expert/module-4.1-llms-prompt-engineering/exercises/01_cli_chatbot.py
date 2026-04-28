"""
Exercise 1 — Streaming CLI chatbot

A multi-turn chat in your terminal, with:
  • Streaming output (you see tokens as they arrive).
  • Conversation memory (the model sees the history each turn).
  • A `/reset` command to clear history.
  • A `/cost` command to print accumulated token usage.
  • Graceful Ctrl-C handling.

Run: python exercises/01_cli_chatbot.py     # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic

# Make the parent module importable for shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Keep replies under 4 sentences unless the user asks for more."
)


def main():
    require_api_key()
    client = anthropic.Anthropic()

    history: list[dict] = []
    totals = {"input": 0, "output": 0, "cost": 0.0}

    print("Streaming CLI chatbot. Commands: /reset  /cost  /quit")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        if user == "/cost":
            print(f"  input tokens:  {totals['input']}")
            print(f"  output tokens: {totals['output']}")
            print(f"  estimated $:   ${totals['cost']:.5f}")
            continue

        history.append({"role": "user", "content": user})
        print("\nclaude> ", end="", flush=True)

        try:
            with client.messages.stream(
                model=DEFAULT_MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=history,
            ) as stream:
                for delta in stream.text_stream:
                    print(delta, end="", flush=True)
                final = stream.get_final_message()
        except anthropic.APIStatusError as e:
            print(f"\n(API error: {e})")
            history.pop()
            continue

        # Record assistant message in history & track cost
        text = "".join(b.text for b in final.content if b.type == "text")
        history.append({"role": "assistant", "content": text})

        totals["input"]  += final.usage.input_tokens or 0
        totals["output"] += final.usage.output_tokens or 0
        totals["cost"]   += cost_of_usage(final.usage)

        print()  # newline after the streamed reply

    print(f"\nbye! total cost ≈ ${totals['cost']:.5f}")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add prompt caching for the system prompt once the chat is long enough.
# • Persist history to a JSON file; reload on next run.
# • Add a `/save FILENAME` command to dump the chat to disk.
# • Add a `/model NAME` command to swap between haiku / sonnet / opus.
# ─────────────────────────────────────────────────────────────────────────────
