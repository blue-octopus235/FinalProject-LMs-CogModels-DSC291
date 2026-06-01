"""Train one LSTM LM on one noise condition.

Example (Colab T4, full-ish run):
    python src/train.py --condition baseline --max_sentences 150000 \
        --epochs 5 --seed 1 --data_dir data --out_dir checkpoints

Holds out the fixed test indices from training (leak-free). Vocab is shared:
built once from the ORIGINAL file and cached at <out_dir>/vocab.txt so every
condition and the eval use an identical vocabulary.
"""
import argparse
import functools
import os
import time

import torch
import torch.nn as nn

from data_utils import (CONDITIONS, ORIGINAL_FILE, build_vocab, collate,
                        get_test_indices, pick_device, read_corpus_column,
                        LMDataset, Vocab)
from lstm_lm import LSTMLanguageModel


def save_checkpoint(model, args, rate, path, step=None):
    payload = {"model_state": model.state_dict(), "args": vars(args), "rate": rate}
    if step is not None:
        payload["step"] = step
    torch.save(payload, path)


def get_or_build_vocab(data_dir, out_dir, max_size, max_sentences_for_vocab=None):
    vocab_path = os.path.join(out_dir, "vocab.txt")
    if os.path.exists(vocab_path):
        return Vocab.load(vocab_path)
    print("[vocab] building shared vocab from original file ...")
    toks = read_corpus_column(os.path.join(data_dir, ORIGINAL_FILE),
                              column="orig_sentence",
                              max_sentences=max_sentences_for_vocab)
    vocab = build_vocab(toks, max_size=max_size)
    vocab.save(vocab_path)
    print(f"[vocab] size={len(vocab)} saved to {vocab_path}")
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--max_sentences", type=int, default=150000,
                    help="cap training size; KEEP CONSTANT across conditions")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt_every_steps", type=int, default=0,
                    help="also save intermediate checkpoints every N optimizer "
                         "steps for learning-trajectory analysis "
                         "(0 = off; final checkpoint is always saved)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()
    print(f"[train] condition={args.condition} seed={args.seed} device={device}")

    test_idx = get_test_indices()
    vocab = get_or_build_vocab(args.data_dir, args.out_dir, args.vocab_size)

    fname, rate = CONDITIONS[args.condition]
    print(f"[data] reading {fname} (excluding {len(test_idx)} test rows) ...")
    train_toks = read_corpus_column(os.path.join(args.data_dir, fname),
                                    column="orig_sentence",
                                    exclude_indices=test_idx,
                                    max_sentences=args.max_sentences)
    print(f"[data] train sentences: {len(train_toks)}")
    ds = LMDataset(train_toks, vocab)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=functools.partial(collate, pad_id=vocab.pad))

    model = LSTMLanguageModel(len(vocab), args.emb_dim, args.hidden_dim,
                              args.n_layers, args.dropout, pad_id=vocab.pad,
                              tie_weights=(args.emb_dim == args.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss, total_tok = 0.0, 0
        for inp, tgt, _ in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            logits, _ = model(inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            global_step += 1
            if args.ckpt_every_steps and global_step % args.ckpt_every_steps == 0:
                step_ckpt = os.path.join(
                    args.out_dir,
                    f"lstm_{args.condition}_seed{args.seed}_step{global_step}.pt")
                save_checkpoint(model, args, rate, step_ckpt, step=global_step)
                print(f"[train] saved trajectory checkpoint {step_ckpt}")
                model.train()
            ntok = (tgt != vocab.pad).sum().item()
            total_loss += loss.item() * ntok
            total_tok += ntok
        ppl = torch.exp(torch.tensor(total_loss / max(total_tok, 1)))
        print(f"[epoch {epoch}] train ppl={ppl:.2f}  ({time.time()-t0:.1f}s)")

    ckpt = os.path.join(args.out_dir, f"lstm_{args.condition}_seed{args.seed}.pt")
    save_checkpoint(model, args, rate, ckpt, step=global_step)
    print(f"[train] saved {ckpt}")


if __name__ == "__main__":
    main()
