"""
Data loading, vocabulary, and the leak-free train/test split.

Design (see README):
  * Kelly's 5 corrupted corpora are ROW-ALIGNED with the original
    agr_50_mostcommon_10K.tsv (identical line counts, no shuffle on her side).
  * We therefore choose a fixed set of TEST ROW INDICES once, exclude those
    rows from every training corpus, and build the minimal-pair test set from
    the ORIGINAL file at those same indices (where we also get verb_index and
    n_diff_intervening for the attractor analysis).
  * Only the training-noise level varies across conditions; the held-out test
    set is identical for all of them.

We train the LM on the `orig_sentence` column (real, vocab-controlled words;
rare tokens already anonymized to a0/a1/... by the Linzen preprocessing).
The `sentence` (POS-substituted) column is NOT used for LM training.
"""
import csv
import os
import sys

import numpy as np
import torch

# ----- special tokens -----
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# csv field-size limit (some rows are long)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CONDITIONS = {
    "baseline":    ("sva_corpus_baseline_rate_0.0.tsv",      0.0),
    "low":         ("sva_corpus_low_rate_0.004.tsv",         0.004),
    "medium_low":  ("sva_corpus_medium_low_rate_0.02.tsv",   0.02),
    "medium_high": ("sva_corpus_medium_high_rate_0.1.tsv",   0.10),
    "high":        ("sva_corpus_high_rate_0.5.tsv",          0.50),
}
ORIGINAL_FILE = "agr_50_mostcommon_10K.tsv"
N_ROWS = 1577211  # data rows in every aligned file


def get_test_indices(n_test=20000, seed=1234, n_rows=N_ROWS):
    """Fixed, seeded set of held-out row indices (0-based into the data rows)."""
    rng = np.random.RandomState(seed)
    return set(rng.choice(n_rows, size=n_test, replace=False).tolist())


def read_corpus_column(path, column="orig_sentence", exclude_indices=None,
                       max_sentences=None):
    """Stream a TSV, yield token lists for `column`, skipping exclude_indices.

    Returns a list of token-lists. Memory: ~ a few GB for the full 1.5M rows;
    use max_sentences to cap for local smoke tests.
    """
    exclude = exclude_indices or set()
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i in exclude:
                continue
            text = row.get(column) or ""
            toks = text.split()
            if toks:
                out.append(toks)
            if max_sentences is not None and len(out) >= max_sentences:
                break
    return out


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for tok, idx in stoi.items():
            self.itos[idx] = tok
        self.pad = stoi[PAD]
        self.bos = stoi[BOS]
        self.eos = stoi[EOS]
        self.unk = stoi[UNK]

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens, add_bos=True, add_eos=True):
        ids = [self.bos] if add_bos else []
        ids += [self.stoi.get(t, self.unk) for t in tokens]
        if add_eos:
            ids.append(self.eos)
        return ids

    def has(self, token):
        return token in self.stoi

    def save(self, path):
        with open(path, "w") as f:
            for tok in self.itos:
                f.write(tok + "\n")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            itos = [line.rstrip("\n") for line in f]
        return cls({t: i for i, t in enumerate(itos)})


def build_vocab(token_lists, max_size=10000, min_freq=1):
    from collections import Counter
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    stoi = {t: i for i, t in enumerate(SPECIALS)}
    for tok, c in counter.most_common():
        if len(stoi) >= max_size:
            break
        if c < min_freq or tok in stoi:
            continue
        stoi[tok] = len(stoi)
    return Vocab(stoi)


class LMDataset(torch.utils.data.Dataset):
    """Per-sentence LM dataset. Sentences in this corpus are independent,
    so we do NOT stream across sentence boundaries."""

    def __init__(self, token_lists, vocab):
        self.vocab = vocab
        self.data = [vocab.encode(toks) for toks in token_lists]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.long)


def collate(batch, pad_id):
    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
    maxlen = int(lengths.max())
    padded = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : len(x)] = x
    # inputs = all but last, targets = all but first
    return padded[:, :-1], padded[:, 1:], lengths - 1
