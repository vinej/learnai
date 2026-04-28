"""
05 — Alignment: SFT, RLHF, DPO

After pre-training, base LLMs predict next tokens — they aren't yet aligned
to what humans want. The alignment pipeline:

  1. SFT (Supervised Fine-Tune)
        Train on ~50k high-quality (instruction, response) pairs.
        Teaches the model to follow instructions in the right format.
        This is what 95% of "fine-tuning" means in practice.

  2. RLHF (Reinforcement Learning from Human Feedback)
        Step 1: collect (chosen, rejected) PREFERENCE PAIRS.
        Step 2: train a REWARD MODEL on those pairs.
        Step 3: PPO-train the LLM to maximize the reward model's score.
        Powerful but tricky and expensive (3-stage pipeline, many hyperparams).

  3. DPO (Direct Preference Optimization) — the popular replacement
        Skip the reward model. Train DIRECTLY on (chosen, rejected) pairs.
        Loss pushes the policy probability of `chosen` UP and `rejected` DOWN
        relative to a frozen reference model.
        Single training step, no PPO, no value function. Much simpler.

  4. Constitutional AI / RLAIF
        Use an LLM to generate AI feedback (a "constitution") instead of
        humans. Cheaper, scalable, often complementary.

This file walks DPO conceptually, with a tiny working example using TRL.

Run: python 05_alignment.py
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "distilgpt2"
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# DPO loss in one short paragraph
# ───────────────────────────────────────────────────────────
DPO_INTUITION = """\
DPO loss for a (prompt, chosen, rejected) triple:

  L = -log σ( β · [ log π_θ(chosen | prompt) - log π_ref(chosen | prompt)
                  - log π_θ(rejected | prompt) + log π_ref(rejected | prompt) ] )

In words: under the trained policy π_θ, the chosen response should be more
likely (relative to the frozen reference π_ref) than the rejected one.
β controls the strength: higher β = trust preferences more.

That's it. No reward model. No PPO. No KL penalty (it's baked in via the
reference model).
"""
print(DPO_INTUITION)


# ───────────────────────────────────────────────────────────
# Tiny preference dataset: prefer concise answers
# ───────────────────────────────────────────────────────────
DATA = [
    {
        "prompt":   "Question: What is the capital of France?\nAnswer:",
        "chosen":   " Paris.",
        "rejected": " The capital of France, a country in Western Europe known for its cuisine, art, and rich history, is in fact the major city of Paris which has long been considered a cultural and political center.",
    },
    {
        "prompt":   "Question: What is the boiling point of water?\nAnswer:",
        "chosen":   " 100 degrees Celsius.",
        "rejected": " Water, an essential compound of two hydrogen atoms and one oxygen atom, boils at a temperature of approximately one hundred degrees Celsius under standard atmospheric pressure conditions at sea level on Earth.",
    },
    {
        "prompt":   "Question: How many continents are there?\nAnswer:",
        "chosen":   " Seven.",
        "rejected": " There are seven continents on the planet Earth, although the exact count can vary depending on the cultural and geographic conventions used in different regions of the world.",
    },
] * 8


dataset = Dataset.from_list(DATA)
print(f"\nDPO dataset: {len(dataset)} preference pairs")


# ───────────────────────────────────────────────────────────
# Models — policy and reference (same checkpoint, ref frozen)
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref    = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
for p in ref.parameters():
    p.requires_grad = False


# ───────────────────────────────────────────────────────────
# Train (very short — just to show the API)
# ───────────────────────────────────────────────────────────
config = DPOConfig(
    output_dir="scratch/dpo-distilgpt2",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    beta=0.1,                       # KL strength — start at 0.1
    max_length=128,
    max_prompt_length=64,
    save_strategy="no",
    logging_steps=5,
    report_to="none",
)
trainer = DPOTrainer(
    model=policy,
    ref_model=ref,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
print("\nrunning a short DPO pass (a few seconds on CPU)...")
trainer.train()
print("done")


# ───────────────────────────────────────────────────────────
# Compare outputs before/after
# ───────────────────────────────────────────────────────────
def gen(model, prompt, max_new=40):
    inp = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


prompt = "Question: What is the speed of light?\nAnswer:"
print("\n=== reference (un-DPOed) ===")
print(repr(gen(ref, prompt)))
print("=== policy (after DPO toward concise answers) ===")
print(repr(gen(policy, prompt)))


# ───────────────────────────────────────────────────────────
# When to use which
# ───────────────────────────────────────────────────────────
ALIGNMENT_GUIDE = """\
USE SFT WHEN
  • You have (instruction, ideal output) pairs.
  • You want the model to follow your task / format.
  • The KEY metric is correctness of the output.

USE DPO WHEN
  • You have (chosen, rejected) preference pairs.
  • The "right" answer is fuzzy — readers prefer A over B subjectively.
  • Tone, helpfulness, safety, conciseness.

USE RLHF (PPO) WHEN
  • You have a reward model (or want to train one) that captures complex
    reward structure (multi-step, multi-objective).
  • You're a research lab or a deeply-resourced team. PPO is hard.

USE CONSTITUTIONAL / RLAIF WHEN
  • You can encode the criteria as RULES and have a strong model evaluate
    against them.
  • Saves human-labeling cost dramatically; quality follows the judge model.

ORDER: usually SFT → (DPO or RLHF) → safety filter / red-team → ship.
"""
print(ALIGNMENT_GUIDE)


# ───────────────────────────────────────────────────────────
# Newer alignment methods you'll see in papers
# ───────────────────────────────────────────────────────────
# • IPO (Identity Preference Optimization) — variant of DPO.
# • KTO (Kahneman-Tversky Optimization) — uses BINARY good/bad signals
#   instead of pairwise preferences. Cheaper to label.
# • ORPO (Odds-Ratio Preference Optimization) — combines SFT and DPO in one step.
# • SimPO — DPO without a reference model.
# All are implemented in TRL with the same SFTTrainer-style API.
