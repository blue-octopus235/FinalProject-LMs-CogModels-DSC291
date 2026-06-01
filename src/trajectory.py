"""Evaluate the learning trajectory of an LSTM across training-step checkpoints.

Answers RQ1 ("does noise *delay* or *prevent* acquisition of the hierarchical
rule") by scoring every intermediate checkpoint that `train.py --ckpt_every_steps`
left behind, not just the final model. For each step we reuse the exact
minimal-pair test set and the same metrics as the one-shot eval
(`evaluate.summarize`): minimal-pair accuracy overall / by attractor, the
attractor gap, and mean surprisal of the correct verb.

Run it (from the repo root) after a trajectory training run:
    python src/train.py --condition baseline --max_sentences 3000 --epochs 2 \
        --seed 1 --ckpt_every_steps 50
    python src/trajectory.py --condition baseline --seed 1

Appends one row per step to results/trajectory_results.csv. Plot acc / gap vs.
step to see whether higher noise shifts the acquisition curve right (delay) or
caps its asymptote (prevention).
"""
import argparse
import csv
import glob
import os
import re

import torch

from data_utils import ORIGINAL_FILE, get_test_indices, pick_device, Vocab
from evaluate import summarize, verb_logprobs
from lstm_lm import LSTMLanguageModel
from minimal_pairs import build_minimal_pairs

STEP_RE = re.compile(r"_step(\d+)\.pt$")


def find_step_checkpoints(out_dir, condition, seed):
    """Return [(step, path), ...] sorted by step for one condition/seed."""
    pattern = os.path.join(out_dir, f"lstm_{condition}_seed{seed}_step*.pt")
    found = []
    for path in glob.glob(pattern):
        m = STEP_RE.search(path)
        if m:
            found.append((int(m.group(1)), path))
    found.sort()
    return found


def load_model(path, vocab, device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = LSTMLanguageModel(len(vocab), a["emb_dim"], a["hidden_dim"],
                              a["n_layers"], a["dropout"], pad_id=vocab.pad,
                              tie_weights=(a["emb_dim"] == a["hidden_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--results_csv", default="results/trajectory_results.csv")
    args = ap.parse_args()

    device = pick_device()
    vocab = Vocab.load(os.path.join(args.out_dir, "vocab.txt"))

    ckpts = find_step_checkpoints(args.out_dir, args.condition, args.seed)
    if not ckpts:
        raise SystemExit(
            f"[trajectory] no step checkpoints found for condition="
            f"{args.condition} seed={args.seed} in {args.out_dir}. "
            f"Train with --ckpt_every_steps > 0 first.")
    print(f"[trajectory] {len(ckpts)} step checkpoints found")

    # Build the minimal-pair test set ONCE (identical for every step).
    test_idx = get_test_indices()
    pairs, skipped = build_minimal_pairs(
        os.path.join(args.data_dir, ORIGINAL_FILE), test_idx, vocab=vocab)
    print(f"[trajectory] usable pairs={len(pairs)}  skipped={skipped}")

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    cols = ["condition", "rate", "seed", "step", "acc_all", "acc_no_attractor",
            "acc_attractor", "attractor_gap", "mean_surprisal_correct_bits",
            "n_all", "n_attractor", "n_no_attractor", "checkpoint"]
    with open(args.results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for step, path in ckpts:
            model, ckpt = load_model(path, vocab, device)
            lp_c, lp_i = verb_logprobs(model, vocab, pairs, device)
            res = summarize(pairs, lp_c, lp_i)
            res.update({"condition": args.condition, "rate": ckpt.get("rate"),
                        "seed": args.seed, "step": step,
                        "checkpoint": os.path.basename(path)})
            w.writerow({k: res.get(k) for k in cols})
            print(f"  step {step:>7}: acc_all={res['acc_all']:.3f}  "
                  f"gap={res['attractor_gap']:.3f}")
    print(f"[trajectory] appended {len(ckpts)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
