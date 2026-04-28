"""
05 — Structured outputs

Most production LLM features need MACHINE-PARSABLE output, not prose. Three
techniques in increasing reliability:

  1. ASK FOR JSON, parse it. Easy, sometimes flaky.
  2. ASK FOR JSON + Pydantic validation, retry on failure.
  3. USE TOOL CALLING — model emits a typed function call. Most reliable.
     (See file 07 for tool calling proper.)

This file shows (1) and (2). Tool-calling for structured output is a
strong default in production.

Run: python 05_structured_outputs.py    # requires ANTHROPIC_API_KEY
"""

import json

import anthropic
from pydantic import BaseModel, Field, ValidationError

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Define the desired output shape with Pydantic
# ───────────────────────────────────────────────────────────
class Contact(BaseModel):
    name: str
    email: str | None = Field(default=None, description="email if mentioned")
    phone: str | None = Field(default=None, description="phone if mentioned")
    role: str | None = Field(default=None, description="job title or role")


class Extraction(BaseModel):
    contacts: list[Contact]
    summary: str = Field(description="one sentence overall summary")


# Auto-generate the JSON schema from the model
SCHEMA = Extraction.model_json_schema()


SYSTEM_PROMPT = f"""\
You extract contact information from text. Return ONLY a JSON object that
strictly conforms to this schema:

{json.dumps(SCHEMA, indent=2)}

Rules:
- Output JSON only. No prose, no Markdown fences, no preamble.
- Use null for any field you cannot find with high confidence.
- The `summary` is one short sentence describing the document.
"""


SAMPLE_DOC = """\
Hi team,

For the new project we'll need to coordinate with two contacts:
- Sarah Chen (sarah.chen@acme.io), VP of Engineering at Acme
- Mike Patel — phone is (555) 123-4567. He runs procurement at Globex.

Looping in: Jamal Williams (no email yet), our internal PM.

Best,
"""


# ───────────────────────────────────────────────────────────
# Approach 1 — ask for JSON, parse with try/except
# ───────────────────────────────────────────────────────────
def call_for_json(text: str) -> str:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        temperature=0,                   # deterministic for structured output
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    print_usage("extract", resp.usage)
    return resp.content[0].text


raw = call_for_json(SAMPLE_DOC)
print("=== raw model output ===")
print(raw)


# ───────────────────────────────────────────────────────────
# Approach 2 — validate with Pydantic, retry once on failure
# ───────────────────────────────────────────────────────────
def extract_with_retry(text: str, max_attempts: int = 2) -> Extraction:
    error: str | None = None
    for attempt in range(max_attempts):
        prompt = text
        if error:
            prompt += (
                f"\n\nYour previous JSON failed validation: {error}\n"
                "Output ONLY the corrected JSON now."
            )
        raw = call_for_json(prompt)
        try:
            data = json.loads(raw)
            return Extraction.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            error = str(e)
            print(f"\nattempt {attempt + 1} failed: {error}")
    raise RuntimeError(f"failed after {max_attempts} attempts: {error}")


print("\n=== validated extraction ===")
result = extract_with_retry(SAMPLE_DOC)
print(result.model_dump_json(indent=2))


# ───────────────────────────────────────────────────────────
# Why temperature = 0 for structured output
# ───────────────────────────────────────────────────────────
# Sampling diversity is a feature for creative writing, a bug for parsable
# output. T=0 makes the model pick the most-likely token at each step,
# producing the most consistent format.


# ───────────────────────────────────────────────────────────
# Tool-calling alternative (preview — full treatment in file 07)
# ───────────────────────────────────────────────────────────
# Tool calling gives the cleanest structured output because the model emits
# arguments matching a JSON schema directly, and the SDK gives you typed access.
# In practice for "I need a structured object", tool calling is the most reliable:
#
#   tools = [{
#       "name": "extract_contacts",
#       "description": "Extract structured contact info.",
#       "input_schema": SCHEMA,
#   }]
#   resp = client.messages.create(model=..., tools=tools, tool_choice={"type": "tool", "name": "extract_contacts"}, ...)
#   tool_use = next(b for b in resp.content if b.type == "tool_use")
#   parsed = Extraction.model_validate(tool_use.input)


# ───────────────────────────────────────────────────────────
# Production tips
# ───────────────────────────────────────────────────────────
# 1. ALWAYS validate with Pydantic — never trust raw JSON from an LLM.
# 2. INCLUDE the schema in the prompt — small models follow it more reliably.
# 3. RETRY ONCE with the validation error message — catches most fixable mistakes.
# 4. For very strict outputs, use TOOL CALLING (file 07) instead of free-form JSON.
# 5. Log every malformed output — they reveal prompt weaknesses.
