"""
Exercise 3 — Structured extraction with Pydantic validation

Extract a typed object from a fuzzy email-style document. Use TOOL CALLING
(strongest reliability) and validate with Pydantic.

The script asserts:
  - the extraction parses successfully on the first try.
  - all expected entities are present (correct names + at least 2 emails).

Run: python exercises/03_structured_extraction.py     # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import DEFAULT_MODEL, print_usage, require_api_key


# ───────────────────────────────────────────────────────────
# The schema we want
# ───────────────────────────────────────────────────────────
class Person(BaseModel):
    name: str
    email: str | None = None
    role: str | None = None


class Project(BaseModel):
    name: str
    deadline: str | None = Field(default=None, description="ISO date if mentioned")


class Extraction(BaseModel):
    project: Project
    people: list[Person]
    summary: str


SCHEMA = Extraction.model_json_schema()


# ───────────────────────────────────────────────────────────
# The document
# ───────────────────────────────────────────────────────────
DOC = """\
Subject: Kickoff — Project Periwinkle

Team,

We're starting Project Periwinkle, due 2026-06-15. Roles:
- Sarah Chen (sarah.c@acme.io) leads engineering.
- Mike Patel <mpatel@acme.io> is product owner.
- Jamal Williams handles design (no email yet, ping me).

The goal is to ship the new dashboard by deadline.

— Lin
"""


def main():
    require_api_key()
    client = anthropic.Anthropic()

    # Define the function as a tool. Forcing tool_choice = THIS tool makes
    # the model emit input matching the schema.
    tools = [{
        "name": "submit_extraction",
        "description": "Submit the extracted structured data.",
        "input_schema": SCHEMA,
    }]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        temperature=0,
        tools=tools,
        tool_choice={"type": "tool", "name": "submit_extraction"},
        system=(
            "Extract the structured information from the user's message. "
            "Use null for unknown fields. The summary is one short sentence."
        ),
        messages=[{"role": "user", "content": DOC}],
    )
    print_usage("extract", response.usage)

    # Get the tool_use block
    tool_uses = [b for b in response.content if b.type == "tool_use"]
    assert len(tool_uses) == 1, "expected exactly one tool_use call"
    raw = tool_uses[0].input

    print("\n=== raw tool input ===")
    print(raw)

    # Validate with Pydantic — should pass first try
    extraction = Extraction.model_validate(raw)
    print("\n=== validated ===")
    print(extraction.model_dump_json(indent=2))

    # ── Spot checks ──
    names = {p.name for p in extraction.people}
    assert "Sarah Chen" in names
    assert "Mike Patel" in names
    emails = {p.email for p in extraction.people if p.email}
    assert len(emails) >= 2, f"expected ≥ 2 emails, got {emails}"
    assert "Periwinkle" in extraction.project.name
    print("\nstructured extraction passed validation ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Run the same task WITHOUT tool calling (free-form JSON) and compare
#   how often validation fails.
# • Add a more complex schema: nested Departments and Tasks.
# • Build a "review and correct" loop: after extraction, ask the model to
#   double-check its own output.
# ─────────────────────────────────────────────────────────────────────────────
