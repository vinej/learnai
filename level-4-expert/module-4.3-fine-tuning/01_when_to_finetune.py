"""
01 — When to fine-tune (and when NOT to)

Fine-tuning is glamorous but rarely the right first move. The decision tree:

  Is the issue ACCURACY ON KNOWN FACTS?
      → use RAG (Module 4.2). Retrieval-grounded answers beat fine-tuning
        on factual tasks because facts shift; retrieval keeps them fresh.

  Is the issue STYLE / FORMAT?
      → start with PROMPTING (system prompts, few-shot, structured output).
        90% of "make it sound like our brand" is solved by prompts.

  Is the issue you NEED A SPECIFIC OUTPUT FORMAT THAT THE MODEL CAN'T HOLD?
      → fine-tune for FORMAT (small data set: 200-2000 examples works).
        SFT-on-format is the cheapest, highest-ROI fine-tune.

  Is the issue LATENCY / COST AT SCALE?
      → fine-tune a SMALL model on outputs from a big model (distillation).
        Replace 90% of Opus calls with a fine-tuned Haiku-class model.

  Is the issue NEW SKILLS / NEW DOMAIN VOCABULARY?
      → fine-tune SFT on domain examples + RAG. Combine both — fine-tuning
        teaches behavior; RAG teaches facts.

  Is the issue HUMAN-PREFERENCE ALIGNMENT?
      → DPO / RLHF (file 05). You need 1000+ human preference pairs.

Run: python 01_when_to_finetune.py
"""


SCENARIOS = [
    ("My chatbot keeps using markdown bullet points; I want plain prose.",
     "Prompt engineering (file 4.1.04). Add 'No bullet points; use prose.' to the system prompt."),
    ("It hallucinates dates of events from 2024.",
     "RAG (Module 4.2) — retrieve docs that contain accurate dates."),
    ("I need it to ALWAYS output JSON matching a schema.",
     "Tool calling (file 4.1.07) or fine-tune for FORMAT (small SFT)."),
    ("Our company has a unique tone of voice; the LLM doesn't capture it.",
     "Few-shot first; if that's not enough, SFT on 200-1000 in-house examples."),
    ("Inference cost is killing us — we make 10M Opus calls / day.",
     "Distill: fine-tune Haiku on Opus outputs. Often 90% Opus quality at 5% cost."),
    ("Doctors find the answers technically correct but not clinical enough.",
     "SFT on cleaned clinical Q&A + RAG over your medical KB. Combined."),
    ("Our users prefer style A over style B; can the model learn that?",
     "DPO / RLHF — collect preference pairs and train alignment (file 05)."),
    ("We need real-time facts (today's stock price).",
     "Tool use (file 4.1.07). Don't fine-tune; give the model a tool."),
]

print("=== When to fine-tune ===\n")
for scenario, recommendation in SCENARIOS:
    print(f"• {scenario}")
    print(f"  → {recommendation}\n")


# ───────────────────────────────────────────────────────────
# Cost-benefit reality check
# ───────────────────────────────────────────────────────────
COST_NOTES = """\
WHAT FINE-TUNING REALLY COSTS
  - Engineering time      : 2-8 weeks for a real first deployment.
  - Compute               : ~$10 (LoRA on 7B) to ~$10k+ (full FT on 70B+ for SOTA).
  - Data collection       : usually the longest pole. 1k clean examples can take days.
  - Maintenance overhead  : every model upgrade needs to be re-fine-tuned.
  - Eval infrastructure   : a fine-tune without an eval is faith, not engineering.

QUICK-WINS BEFORE YOU CONSIDER IT
  - Better prompts                 (free, hours)
  - Few-shot examples              (free, hours)
  - Structured outputs / tool use  (free, hours)
  - RAG on your data               (~1-2 days for v1)
  - Bigger model                   (Haiku → Sonnet → Opus, hours)
  - Prompt caching                 (free, ~minutes; 90% input cost cut)

WHEN FINE-TUNING IS UNAVOIDABLE
  - Strict output format requirements that prompting can't reliably hold.
  - Latency / cost economics fail at the model size you need.
  - Closed-domain task where the base model lacks vocabulary or style.
  - Human-preference alignment for a deeply subjective task.
"""
print(COST_NOTES)
