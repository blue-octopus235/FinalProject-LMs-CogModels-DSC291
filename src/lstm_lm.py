"""Word-level LSTM language model (Gulordava et al. 2018 / Linzen et al. 2016 style).

Small enough to train all 5 noise conditions on a Colab T4. Hyperparameters are
configurable; defaults are a scaled-down Gulordava setup that learns SVA quickly.
"""
import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, n_layers=2,
                 dropout=0.2, pad_id=0, tie_weights=True):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                            dropout=dropout if n_layers > 1 else 0.0,
                            batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            if hidden_dim != emb_dim:
                raise ValueError("tie_weights requires hidden_dim == emb_dim")
            self.decoder.weight = self.embedding.weight
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x, hidden=None):
        emb = self.drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.drop(out)
        logits = self.decoder(out)
        return logits, hidden
