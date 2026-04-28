"""
01 — Tool design

Tools are typed functions you give the model. Three rules of thumb:

  1. Each tool does ONE thing. The description fits in two sentences.
  2. Inputs are TIGHTLY SPECIFIED — enums, required fields, bounded numbers.
  3. Outputs are STRUCTURED — typically JSON the model can parse if needed.

Bad tools are harder to fix than bad prompts. Spend time on the schema.

Run: python 01_tool_design.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field, ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# A bad tool — vague and dangerous
# ───────────────────────────────────────────────────────────
BAD_TOOL = {
    "name": "do_thing",
    "description": "Does something with stuff.",
    "input_schema": {
        "type": "object",
        "properties": {"args": {"type": "string"}},
    },
}


# ───────────────────────────────────────────────────────────
# A good tool — tight, narrow, validated
# ───────────────────────────────────────────────────────────
GOOD_TOOL = {
    "name": "schedule_reminder",
    "description": (
        "Schedule a reminder for the user at a specific time. Returns the "
        "reminder ID. Use ONLY when the user asks to be reminded; do not "
        "schedule reminders without the user's explicit request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The reminder text. ≤ 140 characters.",
                "maxLength": 140,
            },
            "when_iso": {
                "type": "string",
                "description": (
                    "ISO 8601 datetime (UTC). Must be in the future. "
                    "Example: 2026-04-30T10:00:00Z."
                ),
                "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
            },
            "channel": {
                "type": "string",
                "enum": ["email", "sms", "push"],
                "description": "Where to deliver the reminder.",
                "default": "email",
            },
        },
        "required": ["message", "when_iso"],
    },
}


# ───────────────────────────────────────────────────────────
# Validate the tool's INPUT with Pydantic when called
# ───────────────────────────────────────────────────────────
# JSON schema in the request reduces malformed calls but doesn't catch all
# domain rules. Pydantic gives you a real second line of defense.
class ReminderInput(BaseModel):
    message: str = Field(min_length=1, max_length=140)
    when_iso: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    channel: str = Field(default="email", pattern=r"^(email|sms|push)$")


def schedule_reminder(args: dict) -> str:
    """Local implementation. Validates input rigorously."""
    try:
        parsed = ReminderInput.model_validate(args)
    except ValidationError as e:
        return json.dumps({"error": "invalid_input", "details": e.errors()})

    reminder_id = "rem_abc123"      # pretend we created one
    return json.dumps({
        "reminder_id": reminder_id,
        "message":  parsed.message,
        "when_iso": parsed.when_iso,
        "channel":  parsed.channel,
    })


# ───────────────────────────────────────────────────────────
# Run a single turn — see how the model uses the tool
# ───────────────────────────────────────────────────────────
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=400,
    tools=[GOOD_TOOL],
    messages=[{"role": "user",
               "content": "Remind me at 9am tomorrow UTC to call Sarah. Use email."}],
)
print_usage("turn 1", response.usage)

# The model should issue a tool_use block. If it does:
for block in response.content:
    if block.type == "tool_use":
        print(f"\nmodel called: {block.name}")
        print(f"args: {block.input}")
        result = schedule_reminder(block.input)
        print(f"local impl returned: {result}")


# ───────────────────────────────────────────────────────────
# What if you skip the schema (BAD_TOOL above)?
# ───────────────────────────────────────────────────────────
# • The model has to guess what `args` should look like.
# • Different runs produce different shapes; downstream parsing breaks.
# • It will frequently use the tool when it shouldn't, because the
#   description doesn't say when to use it.
# Tight schemas + clear descriptions are NOT cosmetic — they materially
# improve reliability.


# ───────────────────────────────────────────────────────────
# Tool DESIGN principles
# ───────────────────────────────────────────────────────────
PRINCIPLES = """\
NAMING
  • Use VERB_NOUN. `schedule_reminder`, `read_file`, `run_sql_query`.
  • Avoid acronyms unless universally known.

DESCRIPTION
  • Two sentences. Sentence 1: what it does + return shape.
                   Sentence 2: when to use / not use.
  • Always include "Use ONLY when ..." or "Do NOT use for ...".

PARAMETERS
  • One parameter = one job. Don't pack multiple inputs into a string.
  • Use `enum` for fixed sets. Pattern/format for strings.
  • `required` only what's truly required; sensible defaults for the rest.
  • Always describe each parameter; the model reads them.

OUTPUTS
  • Return JSON for downstream tools. Wrap in {"error": ...} on failure.
  • Return SHORT plain text only when there's no chance of further chaining.
  • Never raise: catch and return an error object.

PERMISSIONS
  • Mark which tools modify state (write_file, delete_record).
  • Gate write tools behind explicit user confirmation in production.
"""
print(PRINCIPLES)
