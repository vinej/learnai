# Module 4.1 — LLMs & Prompt Engineering

**Level:** 4 — Expert
**Estimated time:** 2 weeks

## Goal
Use large language models effectively and economically through their APIs.

## Topics
### How LLMs work (engineer's view)
- Decoder-only transformers, autoregressive generation
- Tokenizers and why token counts matter (cost & context)
- Sampling: temperature, top-k, top-p, repetition penalty
- Context windows, attention cost (quadratic)

### Prompt engineering
- System / user / assistant roles
- Zero-shot, few-shot, chain-of-thought, self-consistency
- Structured outputs (JSON mode, schema-constrained generation)
- Delimiters, role-playing, persona prompts
- Common failure modes: hallucinations, refusal, format drift

### Working with the Anthropic Python SDK
- Auth, models, basic completions
- Streaming responses
- **Prompt caching** (huge cost saver — required for production)
- Vision inputs, PDFs, files API
- Extended thinking
- Tool use / function calling
- Token & cost accounting

### Cross-provider patterns
- OpenAI, Anthropic, Google, open-source via vLLM/Ollama
- Abstractions: when to use LiteLLM vs SDKs directly

## Exercises
1. Build a CLI chatbot using the Anthropic Python SDK with streaming.
2. Refactor it to use prompt caching for a large system prompt; verify cache hits in usage.
3. Force structured JSON output for a data-extraction task; validate with Pydantic.
4. Run the same prompt across 3 temperature settings and analyze output diversity.
5. Implement a few-shot classifier and compare it to a fine-tuned BERT from Module 3.3.

## Resources
- Anthropic API docs: https://docs.anthropic.com
- Anthropic prompt-engineering guides
- "Prompt Engineering Guide" — DAIR.AI
- Lilian Weng — "Prompt Engineering" blog post

## Checkpoint
You can build a production LLM call: streaming, cached, with structured output, with token usage and cost tracked per request.
