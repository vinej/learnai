"""
08 — Agent frameworks: the buffet

You've now built agent loops by hand. Several frameworks try to do that
work for you. The tradeoffs are similar to RAG frameworks (Module 4.2.08):

  EASY DEMO          ↔  HEAVY ABSTRACTION YOU MUST DEBUG INTO
  PREBUILT PATTERNS  ↔  FEWER DEGREES OF FREEDOM
  ACTIVE COMMUNITY   ↔  RAPID API CHURN

Run: python 08_frameworks.py
"""


# ───────────────────────────────────────────────────────────
# 1. Claude Agent SDK
# ───────────────────────────────────────────────────────────
# Anthropic's official SDK for building agents on top of Claude Code's
# harness. You get sessions, tools (including powerful built-ins like
# Bash, Read, Write, Edit, Grep, Glob), and conversation management.
#
# Best for: agents that USE THE FILESYSTEM, RUN COMMANDS, or build code.
CLAUDE_AGENT_SDK = """\
# pip install claude-agent-sdk

from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a careful refactoring assistant.",
        # The harness already ships strong tools; you can add custom ones via MCP.
        cwd="/path/to/repo",
    )
    async for message in query(
        prompt="Add type hints to all functions in src/main.py.",
        options=options,
    ):
        print(message)

# Run it:
#   import asyncio; asyncio.run(main())
"""


# ───────────────────────────────────────────────────────────
# 2. LangGraph
# ───────────────────────────────────────────────────────────
# Graph-based agent orchestration from the LangChain team. You declare nodes
# (functions) and edges (transitions). State threads through the graph.
#
# Best for: complex MULTI-STEP workflows where the structure matters
# (planner → researcher → critic → writer). Built-in checkpointing,
# time-travel, human-in-the-loop interrupts.
LANGGRAPH = """\
# pip install langgraph langchain-anthropic

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    question: str
    plan: str
    answer: str

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def plan(state: State):
    prompt = f"Plan steps to answer: {state['question']}"
    return {"plan": llm.invoke(prompt).content}

def execute(state: State):
    prompt = f"Question: {state['question']}\\nPlan: {state['plan']}\\nAnswer:"
    return {"answer": llm.invoke(prompt).content}

graph = StateGraph(State)
graph.add_node("plan",    plan)
graph.add_node("execute", execute)
graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)
app = graph.compile()

result = app.invoke({"question": "Why did transformers replace LSTMs?"})
"""


# ───────────────────────────────────────────────────────────
# 3. CrewAI
# ───────────────────────────────────────────────────────────
# Role-based multi-agent orchestration. You define "agents" and "tasks";
# the framework runs them sequentially or hierarchically.
CREWAI = """\
# pip install crewai
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Gather facts", backstory="...", tools=[...])
writer     = Agent(role="Writer",     goal="Craft summary", backstory="...")

t1 = Task(description="Research transformers", agent=researcher)
t2 = Task(description="Write a 1-paragraph summary", agent=writer, context=[t1])

crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
result = crew.kickoff()
"""


# ───────────────────────────────────────────────────────────
# 4. AutoGen / smolagents / others
# ───────────────────────────────────────────────────────────
# AutoGen — Microsoft research; multi-agent conversations.
# smolagents — HuggingFace's tiny agent runtime, code-first.
# semantic-kernel — Microsoft; agent-friendly orchestration.


# ───────────────────────────────────────────────────────────
# 5. Build it yourself
# ───────────────────────────────────────────────────────────
# What you've done in files 02 and 07. Honestly, for many production
# agents the right answer is 100-300 lines of carefully-thought-out
# Python. You debug, you log, you ship.


# ───────────────────────────────────────────────────────────
# Picking guide
# ───────────────────────────────────────────────────────────
DECISION = """\
WANT TO BUILD A CODE / FILE / SHELL AGENT
  → Claude Agent SDK. The built-in tools are tested and powerful.

NEED EXPLICIT GRAPH OF STEPS WITH HUMAN-IN-THE-LOOP CHECKPOINTS
  → LangGraph. It's designed for this.

PROTOTYPING ROLE-BASED MULTI-AGENT
  → CrewAI. Easy to spin up; will lock you into its abstractions.

SHIPPING PRODUCTION FEATURES WITH TIGHT CONTROL
  → Roll your own loop (file 02). Wrap with Langfuse / your own tracer.

BUILDING A RESEARCH / DEMO PROJECT
  → Pick whichever you like and keep moving. Don't rewrite when you find it
    limiting; that's a sign you've graduated to a custom loop.
"""


for label, code in [
    ("Claude Agent SDK",  CLAUDE_AGENT_SDK),
    ("LangGraph",          LANGGRAPH),
    ("CrewAI",             CREWAI),
]:
    print(f"\n=== {label} ===")
    print(code)

print(DECISION)


# ───────────────────────────────────────────────────────────
# MCP — Model Context Protocol
# ───────────────────────────────────────────────────────────
MCP_NOTE = """\
MCP is an open standard for connecting LLMs to data and tools (think:
"USB-C for AI"). An MCP SERVER exposes tools/resources/prompts; an MCP
CLIENT consumes them. The Claude Agent SDK and many frameworks support
adding MCP servers as tool sources.

Examples:
  • mcp-server-filesystem
  • mcp-server-github
  • mcp-server-slack
  • mcp-server-postgres
  Add via:  --mcp-config mcp.json     (Claude Code-style)

If you're building tools that multiple agent frameworks should share,
expose them as an MCP server rather than baking them into one harness.
"""
print(MCP_NOTE)
