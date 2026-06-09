"""Evaluate a trained LSTM on the held-out minimal-pair test set.

Primary metric  : minimal-pair accuracy = fraction of pairs where the model
                  assigns higher P(verb | prefix) to the correct (grammatical)
                  verb form than to the incorrect one.
Secondary       : mean surprisal (-log2 P) of the correct verb.
Rule probing    : accuracy split by attractor presence (n_diff_intervening>0)
                  and the attractor gap = acc(no-attractor) - acc(attractor).
                  A large gap => reliance on the nearest noun (linear rule).

Writes one row to results/eval_results.csv.
"""
import argparse
import csv
import json
import math
import os

import torch

from data_utils import ORIGINAL_FILE, get_test_indices, pick_device, Vocab
from lstm_lm import LSTMLanguageModel
from minimal_pairs import build_minimal_pairs


@torch.no_grad()
def verb_logprobs(model, vocab, pairs, device, batch_size=256):
    """Return (logp_correct, logp_incorrect) arrays over pairs."""
    model.eval()
    lp_c, lp_i = [], []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        # encode prefix with BOS, no EOS; predict next token = the verb
        seqs = [[vocab.bos] + [vocab.stoi.get(t, vocab.unk) for t in p["prefix"]]
                for p in chunk]
        lens = [len(s) for s in seqs]
        maxlen = max(lens)
        inp = torch.full((len(seqs), maxlen), vocab.pad, dtype=torch.long)
        for j, s in enumerate(seqs):
            inp[j, :len(s)] = torch.tensor(s)
        inp = inp.to(device)
        logits, _ = model(inp)                      # (B, T, V)
        last = torch.tensor([l - 1 for l in lens], device=device)
        final_logits = logits[torch.arange(len(seqs), device=device), last]  # (B, V)
        logprobs = torch.log_softmax(final_logits, dim=-1)
        for j, p in enumerate(chunk):
            ci = vocab.stoi[p["correct"]]
            ii = vocab.stoi[p["incorrect"]]
            lp_c.append(logprobs[j, ci].item())
            lp_i.append(logprobs[j, ii].item())
    return lp_c, lp_i


def summarize(pairs, lp_c, lp_i):
    def acc(mask):
        idx = [k for k, m in enumerate(mask) if m]
        if not idx:
            return float("nan"), 0
        correct = sum(1 for k in idx if lp_c[k] > lp_i[k])
        return correct / len(idx), len(idx)

    all_mask = [True] * len(pairs)
    attr = [p["has_attractor"] for p in pairs]
    noattr = [not a for a in attr]
    acc_all, n_all = acc(all_mask)
    acc_attr, n_attr = acc(attr)
    acc_noattr, n_noattr = acc(noattr)
    surprisal = -sum(lp_c) / max(len(lp_c), 1) / math.log(2)
    gap = (acc_noattr - acc_attr) if (n_attr and n_noattr) else float("nan")
    return {
        "acc_all": acc_all, "n_all": n_all,
        "acc_no_attractor": acc_noattr, "n_no_attractor": n_noattr,
        "acc_attractor": acc_attr, "n_attractor": n_attr,
        "attractor_gap": gap,
        "mean_surprisal_correct_bits": surprisal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--results_csv", default="results/eval_results.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--restrict_indices", default=None,
                    help="JSON file with a list of source-row indices. If given, "
                         "score only pairs whose 'idx' is in this set (for the "
                         "matched LSTM-vs-RNNG comparison) and tag rows with a "
                         "'subset' column.")
    ap.add_argument("--subset_label", default="matched",
                    help="Value written to the 'subset' column when "
                         "--restrict_indices is used.")
    args = ap.parse_args()

    device = pick_device()
    vocab = Vocab.load(os.path.join(args.out_dir, "vocab.txt"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt["args"]
    model = LSTMLanguageModel(len(vocab), a["emb_dim"], a["hidden_dim"],
                              a["n_layers"], a["dropout"], pad_id=vocab.pad,
                              tie_weights=(a["emb_dim"] == a["hidden_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_idx = get_test_indices()
    pairs, skipped = build_minimal_pairs(
        os.path.join(args.data_dir, ORIGINAL_FILE), test_idx, vocab=vocab)
    print(f"[eval] usable pairs={len(pairs)}  skipped={skipped}")

    if args.restrict_indices:
        with open(args.restrict_indices) as f:
            keep = set(json.load(f))
        before = len(pairs)
        pairs = [p for p in pairs if p["idx"] in keep]
        print(f"[eval] restricted to {len(pairs)}/{before} pairs in "
              f"{args.restrict_indices} (requested {len(keep)} indices) "
              f"[subset={args.subset_label}]")

    lp_c, lp_i = verb_logprobs(model, vocab, pairs, device)
    res = summarize(pairs, lp_c, lp_i)
    res.update({"condition": args.condition, "rate": ckpt.get("rate"),
                "seed": args.seed, "checkpoint": os.path.basename(args.checkpoint)})
    if args.restrict_indices:
        res["subset"] = args.subset_label
    for k, v in res.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    cols = ["condition", "rate", "seed", "acc_all", "acc_no_attractor",
            "acc_attractor", "attractor_gap", "mean_surprisal_correct_bits",
            "n_all", "n_attractor", "n_no_attractor", "checkpoint"]
    if args.restrict_indices:  # extra column only on matched-subset runs (back-compat)
        cols.append("subset")
    with open(args.results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow({k: res.get(k) for k in cols})
    print(f"[eval] appended to {args.results_csv}")


if __name__ == "__main__":
    main()
