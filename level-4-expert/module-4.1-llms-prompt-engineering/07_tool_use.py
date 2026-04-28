"""
07 — Tool use (function calling)

You declare TOOLS — typed Python functions you want the model to be able to
call. The model decides when to call them; you actually run them and feed
the results back.

The conversation looks like:

  user: "What's the weather in Paris?"
  assistant: <tool_use name="get_weather" input={"city": "Paris"}>
  (your code runs get_weather("Paris") → "18°C, sunny")
  user: <tool_result>"18°C, sunny"</tool_result>
  assistant: "It's 18°C and sunny in Paris."

Run: python 07_tool_use.py    # requires ANTHROPIC_API_KEY
"""

import json
from datetime import datetime

import anthropic

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Define your local tools
# ───────────────────────────────────────────────────────────
def get_weather(city: str, unit: str = "celsius") -> str:
    """Look up the weather. Real impl would call an API; we fake it."""
    fake = {"Paris": (18, "sunny"), "Tokyo": (24, "humid"),
            "London": (12, "rainy")}
    if city not in fake:
        return json.dumps({"error": f"unknown city {city}"})
    temp_c, cond = fake[city]
    temp = temp_c if unit == "celsius" else temp_c * 9 / 5 + 32
    return json.dumps({"temperature": temp, "unit": unit, "condition": cond})


def current_time(timezone: str = "UTC") -> str:
    return json.dumps({"now": datetime.utcnow().isoformat() + "Z", "tz": "UTC"})


LOCAL_TOOLS = {
    "get_weather":  get_weather,
    "current_time": current_time,
}


# ───────────────────────────────────────────────────────────
# Tool SCHEMAS — what the model sees
# ───────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather conditions for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "current_time",
        "description": "Get the current UTC time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "default": "UTC"},
            },
            "required": [],
        },
    },
]


# ───────────────────────────────────────────────────────────
# The agent loop
# ───────────────────────────────────────────────────────────
def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """Run the tool-use loop until the model gives a final text answer."""
    messages = [{"role": "user", "content": user_query}]

    for i in range(max_iterations):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=512,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        print_usage(f"turn {i + 1}", response.usage)

        # If the model decided it's done, return its text.
        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if b.type == "text")

        # Otherwise the model used a tool. Find each tool_use block, run it,
        # and append the result.
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    print(f"  → calling tool {tool_name}({tool_input})")
                    result = LOCAL_TOOLS[tool_name](**tool_input)
                    print(f"    result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop — bail
        raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"agent didn't finish in {max_iterations} iterations")


print("=== single-tool query ===")
answer = run_agent("What's the weather in Tokyo?")
print(f"final answer: {answer.strip()}\n")


print("=== multi-tool query ===")
answer = run_agent(
    "Compare the weather in Paris and London right now. "
    "Also tell me the current UTC time."
)
print(f"final answer: {answer.strip()}")


# ───────────────────────────────────────────────────────────
# Forcing a specific tool
# ───────────────────────────────────────────────────────────
# tool_choice options:
#   {"type": "auto"}                         (default — model decides)
#   {"type": "any"}                          (must use SOME tool)
#   {"type": "tool", "name": "get_weather"}  (must use THIS tool)
# Useful for structured-output extraction (file 05's alternative).


# ───────────────────────────────────────────────────────────
# Tool-use design tips
# ───────────────────────────────────────────────────────────
# • CLEAR DESCRIPTIONS. The model picks a tool based on its description.
#   Spend time on it. Mention parameters and edge cases.
# • TIGHT SCHEMAS. Use enums, required fields, descriptions per parameter.
# • RETURN STRUCTURED RESULTS. JSON strings work. Natural-language results
#   work too. The model handles both — but JSON is parseable downstream.
# • FAIL GRACEFULLY. If your tool errors, return JSON like {"error": "..."}
#   so the model can ask the user for clarification or pick another tool.
# • SECURITY: tools that touch the file system, run code, or call paid APIs
#   should be gated behind permission checks. See Module 4.4 (Agents).
# • LOGGING: log every tool call with arguments and result. Tracing shows
#   how the model is actually using your tools.


# ───────────────────────────────────────────────────────────
# Tool use unlocks "agents"
# ───────────────────────────────────────────────────────────
# Once the model can call functions, you can build:
#   • Search-the-web agents (a `web_search` tool).
#   • Code-running agents (a `python` tool sandboxed in Docker / E2B).
#   • RAG (a `retrieve` tool that hits your vector store).
#   • Multi-agent orchestration (one agent's tool calls another agent).
# Module 4.4 covers building robust, sandboxed agents.
