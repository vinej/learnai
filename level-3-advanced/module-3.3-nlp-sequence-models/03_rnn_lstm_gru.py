"""
03 — RNN, LSTM, GRU

Before transformers, sequences were processed by RECURRENT networks: read
one token, update a hidden state, repeat. The hidden state is a fixed-size
"memory" of everything seen so far.

  RNN   — h_t = tanh(W [x_t, h_{t-1}] + b).
          Vanishing gradients on long sequences.
  LSTM  — adds input/forget/output GATES + a cell state. Much better at
          long-range dependencies.
  GRU   — simplified LSTM (2 gates). Similar performance, fewer params.

Transformers replaced these in most NLP — no recurrence, fully parallel,
arbitrary attention spans. RNN/LSTM/GRU are still useful for small models,
streaming, or hard hardware constraints.

Run: python 03_rnn_lstm_gru.py
"""

import torch
import torch.nn as nn


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# nn.RNN, nn.LSTM, nn.GRU: same API
# ───────────────────────────────────────────────────────────
batch, seq_len, in_dim, hidden = 4, 12, 16, 32
x = torch.randn(batch, seq_len, in_dim)

rnn  = nn.RNN(in_dim,  hidden, batch_first=True)
lstm = nn.LSTM(in_dim, hidden, batch_first=True)
gru  = nn.GRU(in_dim,  hidden, batch_first=True)

# RNN/GRU return: (output_for_each_step, last_hidden)
rnn_out, h_rnn = rnn(x)
print(f"RNN output: {tuple(rnn_out.shape)}, last h: {tuple(h_rnn.shape)}")

# LSTM additionally returns the cell state
lstm_out, (h_lstm, c_lstm) = lstm(x)
print(f"LSTM output: {tuple(lstm_out.shape)}, last h: {tuple(h_lstm.shape)}, cell: {tuple(c_lstm.shape)}")

gru_out, h_gru = gru(x)
print(f"GRU output: {tuple(gru_out.shape)}, last h: {tuple(h_gru.shape)}")


# ───────────────────────────────────────────────────────────
# Stacking and bidirectionality
# ───────────────────────────────────────────────────────────
# num_layers > 1: stacked RNNs (deeper).
# bidirectional=True: read left→right AND right→left, concatenate.
bi_lstm = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True, batch_first=True)
bi_out, (bi_h, bi_c) = bi_lstm(x)
print(f"\nBi-LSTM output: {tuple(bi_out.shape)}  (hidden doubles for bidi)")
# bi_out: (B, T, 2 * hidden). bi_h: (num_layers * 2, B, hidden).


# ───────────────────────────────────────────────────────────
# A complete tiny char model
# ───────────────────────────────────────────────────────────
class CharLSTM(nn.Module):
    """Predict the next character. Standard sequence model template."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                             batch_first=True)
        self.head  = nn.Linear(hidden, vocab_size)

    def forward(self, ids, hidden=None):
        emb = self.embed(ids)                # (B, T, E)
        out, hidden = self.lstm(emb, hidden) # (B, T, H)
        logits = self.head(out)              # (B, T, V)
        return logits, hidden


vocab_size = 30
model = CharLSTM(vocab_size)
ids = torch.randint(0, vocab_size, (4, 20))
logits, hidden = model(ids)
print(f"\nCharLSTM logits shape: {tuple(logits.shape)}")


# ───────────────────────────────────────────────────────────
# Training pattern (sketch)
# ───────────────────────────────────────────────────────────
#   Given input ids x of length T, the targets are x SHIFTED by 1:
#       inputs  = x[:-1]      → predict x[1:]
#   Loss = cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
#
# At inference (text generation):
#   Start with a seed token. Feed it through the model. Sample the next token
#   from the output distribution. Append. Repeat. Pass `hidden` along to keep
#   memory stateful between calls.


# ───────────────────────────────────────────────────────────
# Why LSTMs beat plain RNNs
# ───────────────────────────────────────────────────────────
# Plain RNN gradient through T steps multiplies T weight matrices. If their
# eigenvalues are < 1 → gradient → 0 (vanishing). If > 1 → blows up (exploding).
#
# LSTM's CELL STATE flows mostly ADDITIVELY through time, gated by the
# forget gate. Gradients flow through addition cleanly → much longer
# effective memory.
#
# In practice, LSTMs handle ~100-1000 step dependencies. Beyond that, even
# they struggle. That's why transformers — which see the whole sequence at
# once via attention — took over.


# ───────────────────────────────────────────────────────────
# Practical knobs
# ───────────────────────────────────────────────────────────
# • dropout       — between stacked layers. Use 0.1-0.3.
# • bidirectional — only if you have the FULL sequence (e.g. classifier).
#                   Don't use for autoregressive generation.
# • clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 — RNNs need gradient clipping to avoid explosion.
# • PackedSequence (pack_padded_sequence) for variable-length batching.
