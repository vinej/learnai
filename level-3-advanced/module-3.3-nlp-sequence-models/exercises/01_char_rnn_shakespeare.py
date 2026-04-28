"""
Exercise 1 — Character-level RNN that generates Shakespeare-like text

Train a small LSTM on a few hundred lines of Shakespeare. Sample from it.
This is the canonical "warm fuzzy" deep learning demo.

The script asserts that after training, the model produces text containing
mostly real English-ish characters and spaces (low char-level perplexity).

Run: python exercises/01_char_rnn_shakespeare.py
"""

import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# A few public-domain Shakespeare monologues
# ───────────────────────────────────────────────────────────
TEXT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.
Then the whining school-boy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
"""

# Vocabulary
chars = sorted(set(TEXT))
vocab_size = len(chars)
ch_to_id = {c: i for i, c in enumerate(chars)}
id_to_ch = {i: c for c, i in ch_to_id.items()}
print(f"vocab size: {vocab_size}, text length: {len(TEXT)}")


# Encode the whole corpus
data = torch.tensor([ch_to_id[c] for c in TEXT], dtype=torch.long, device=device)


# ───────────────────────────────────────────────────────────
# Sample fixed-length training chunks
# ───────────────────────────────────────────────────────────
SEQ_LEN = 64
BATCH   = 32


def get_batch():
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([data[s:s + SEQ_LEN]     for s in starts])
    y = torch.stack([data[s + 1:s + SEQ_LEN + 1] for s in starts])
    return x, y


# ───────────────────────────────────────────────────────────
# CharLSTM
# ───────────────────────────────────────────────────────────
class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                             batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.head  = nn.Linear(hidden, vocab_size)

    def forward(self, ids, hidden=None):
        emb = self.embed(ids)
        out, hidden = self.lstm(emb, hidden)
        return self.head(out), hidden


model = CharLSTM(vocab_size).to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

STEPS = 1500
for step in range(STEPS):
    x, y = get_batch()
    optimizer.zero_grad()
    logits, _ = model(x)
    # logits: (B, T, V) → (B*T, V); targets: (B, T) → (B*T,)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 200 == 0 or step == STEPS - 1:
        print(f"step {step:>4}/{STEPS}  loss={loss.item():.3f}")


# ───────────────────────────────────────────────────────────
# Generate
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def generate(seed: str, length: int = 300, temperature: float = 0.8) -> str:
    model.eval()
    ids = torch.tensor([ch_to_id[c] for c in seed], device=device).unsqueeze(0)
    hidden = None
    out_chars = list(seed)

    # Warm up the hidden state with the seed
    logits, hidden = model(ids, hidden)
    next_id = logits[0, -1] / temperature
    next_id = torch.multinomial(torch.softmax(next_id, dim=-1), num_samples=1)

    for _ in range(length):
        out_chars.append(id_to_ch[int(next_id)])
        logits, hidden = model(next_id.view(1, 1), hidden)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

    return "".join(out_chars)


print("\n--- generated sample ---")
sample = generate("To be, or", length=400, temperature=0.7)
print(sample)


# ───────────────────────────────────────────────────────────
# A loose sanity check
# ───────────────────────────────────────────────────────────
real_chars = sum(c.isalpha() or c in " ,.;:?!\n'" for c in sample)
real_rate = real_chars / len(sample)
print(f"\nfraction of 'normal' characters in output: {real_rate:.2%}")
assert real_rate > 0.95, "output should be mostly readable English chars"
print("char-RNN trained ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Increase the corpus 10× by adding more Shakespeare. The output coheres
#   far more.
# • Replace LSTM with a small transformer decoder (you wrote one in file 05!).
#   Compare quality at the same param count.
# • Implement top-k or nucleus (top-p) sampling instead of plain temperature.
# ─────────────────────────────────────────────────────────────────────────────
