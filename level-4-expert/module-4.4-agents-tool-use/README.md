# Module 4.4 — Agents & Tool Use

**Level:** 4 — Expert
**Estimated time:** 2-3 weeks

## Goal
Build LLM systems that can take actions in the world: call tools, browse, write code, and chain steps.

## Topics
### Tool / function calling
- Schema design: clear names, focused descriptions, strict params
- Anthropic tool-use API specifics
- Validation: Pydantic schemas, retries on bad output
- Parallel tool calls

### Agent loops
- The basic loop: think → act → observe → repeat
- ReAct, Reflexion, Plan-and-Execute patterns
- Stop conditions, max iterations, budgets
- Memory: short-term (scratchpad), long-term (vector store)

### Common tools
- Web search & web browsing
- Code execution (sandboxed!)
- File system / RAG retrieval
- API calls (REST, GraphQL)
- Database queries (with safety!)
- MCP (Model Context Protocol) servers

### Multi-agent systems
- Supervisor / worker patterns
- Specialized agents (planner, researcher, critic)
- Communication & shared state
- When multi-agent is overkill (often)

### Frameworks
- **Claude Agent SDK** (`claude-agent-sdk` Python package)
- LangGraph, CrewAI, AutoGen, smolagents
- Building from scratch (often best for production)

### Safety & robustness
- Sandbox: Docker, Firecracker, E2B for code execution
- Prompt injection: defenses (delimiters, ignore-context patterns, output validation)
- Permission boundaries, human-in-the-loop checkpoints
- Cost guards (max tokens, max steps, max wall-clock)
- Logging every tool call for debugging

## Exercises
1. Build a research agent: takes a question, searches the web, synthesizes an answer with citations.
2. Build a SQL agent over a sample database with a read-only sandbox and a query budget.
3. Use the Claude Agent SDK to build an agent that can read/write files in a worktree.
4. Construct a prompt-injection attack on your own agent; then patch it.
5. Add tracing (LangSmith, Langfuse, or your own) so every step is inspectable.

## Resources
- Anthropic agent guides: https://docs.anthropic.com/en/docs/agents-and-tools
- Claude Agent SDK docs
- "Building Effective Agents" — Anthropic blog post
- LangGraph tutorials

## Checkpoint
You can design, build, and operate an agent that uses tools to complete real tasks — with traces, budgets, sandboxing, and prompt-injection defenses.
