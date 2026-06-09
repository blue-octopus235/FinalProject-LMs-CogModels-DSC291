"""H3 significance test: does the RNNG's attractor gap grow more slowly than the LSTM's?

Compares LSTM vs. RNNG on the MATCHED subset (results/matched_idx.json from
src/match_subset.py), per noise condition. Two complementary readouts:

  * Pair-level bootstrap (primary): resample the matched pairs with replacement and
    recompute Δgap = gap_LSTM - gap_RNNG. Reports a 95% CI and a one-sided bootstrap
    p-value (fraction of resamples with Δgap <= 0). Uses thousands of pairs, so it does
    not lean on the 3 seeds. Same for Δacc_all.
  * Seed-level summary (secondary, n=3): mean ± std of each model's gap and the paired
    Δgap across seeds — descriptive context only (small n).

LSTM per-pair correctness is scored on the fly from the committed checkpoints/lstm_*.pt
(reusing evaluate.verb_logprobs), so no extra LSTM eval run is needed. RNNG per-pair
correctness is read from results/rnng_pairs_{cond}_seed{N}.csv (written by
run_rnng_datahub.py).

Run from the repo root:  python src/stats_h3.py
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import torch

from data_utils import ORIGINAL_FILE, get_test_indices, pick_device, Vocab
from evaluate import verb_logprobs
from lstm_lm import LSTMLanguageModel
from minimal_pairs import build_minimal_pairs

CONDS = ["baseline", "low", "medium_low", "medium_high", "high"]
RNNG_RE = re.compile(r"rnng_pairs_(?P<cond>.+)_seed(?P<seed>\d+)\.csv$")


def gap_from_acc(acc_pp, attr):
    """Attractor gap from per-pair accuracy and a boolean attractor mask."""
    return acc_pp[~attr].mean() - acc_pp[attr].mean()


def load_lstm_correct(pairs, out_dir, device):
    """{(cond, seed): bool array over `pairs`} from checkpoints/lstm_*.pt."""
    out = {}
    for ckpt_path in sorted(glob.glob(os.path.join(out_dir, "lstm_*_seed*.pt"))):
        name = os.path.basename(ckpt_path)
        m = re.match(r"lstm_(?P<cond>.+)_seed(?P<seed>\d+)\.pt$", name)
        if not m:
            continue
        cond, seed = m["cond"], int(m["seed"])
        vocab = Vocab.load(os.path.join(out_dir, "vocab.txt"))
        ckpt = torch.load(ckpt_path, map_location=device)
        a = ckpt["args"]
        model = LSTMLanguageModel(
            len(vocab), a["emb_dim"], a["hidden_dim"], a["n_layers"], a["dropout"],
            pad_id=vocab.pad, tie_weights=(a["emb_dim"] == a["hidden_dim"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        lp_c, lp_i = verb_logprobs(model, vocab, pairs, device)
        out[(cond, seed)] = np.array([c > i for c, i in zip(lp_c, lp_i)], dtype=bool)
        print(f"  scored LSTM {cond} seed{seed} ({len(pairs)} pairs)")
    return out


def load_rnng_correct(pairs, pairs_glob):
    """{(cond, seed): bool array over `pairs`} from results/rnng_pairs_*.csv."""
    idx_order = [p["idx"] for p in pairs]
    out = {}
    for path in sorted(glob.glob(pairs_glob)):
        m = RNNG_RE.search(os.path.basename(path))
        if not m:
            continue
        cond, seed = m["cond"], int(m["seed"])
        correct = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                correct[int(r["idx"])] = bool(int(r["correct"]))
        missing = [i for i in idx_order if i not in correct]
        if missing:
            print(f"  WARNING: {cond} seed{seed} missing {len(missing)} matched "
                  f"pairs — skipping this run")
            continue
        out[(cond, seed)] = np.array([correct[i] for i in idx_order], dtype=bool)
    return out


def by_cond(run_dict):
    d = {}
    for (cond, seed), arr in run_dict.items():
        d.setdefault(cond, []).append(arr)
    return d


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--matched_idx", default="results/matched_idx.json")
    ap.add_argument("--rnng_pairs_glob", default="results/rnng_pairs_*.csv")
    ap.add_argument("--out_csv", default="results/h3_stats.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not glob.glob(args.rnng_pairs_glob):
        raise SystemExit(
            f"No RNNG per-pair files at {args.rnng_pairs_glob}. Run "
            f"run_rnng_datahub.py then src/match_subset.py first.")
    if not os.path.exists(args.matched_idx):
        raise SystemExit(f"{args.matched_idx} not found — run src/match_subset.py first.")

    matched = set(json.load(open(args.matched_idx)))
    device = pick_device()
    vocab = Vocab.load(os.path.join(args.out_dir, "vocab.txt"))
    all_pairs, _ = build_minimal_pairs(
        os.path.join(args.data_dir, ORIGINAL_FILE), get_test_indices(), vocab=vocab)
    pairs = [p for p in all_pairs if p["idx"] in matched]
    attr = np.array([p["has_attractor"] for p in pairs], dtype=bool)
    print(f"Matched pairs: {len(pairs)} ({attr.sum()} attractor / {(~attr).sum()} not)\n")

    print("Scoring LSTM checkpoints:")
    lstm = by_cond(load_lstm_correct(pairs, args.out_dir, device))
    print("Loading RNNG per-pair files:")
    rnng = by_cond(load_rnng_correct(pairs, args.rnng_pairs_glob))

    rng = np.random.default_rng(args.seed)
    n = len(pairs)
    rows = []
    for cond in CONDS:
        if cond not in lstm or cond not in rnng:
            print(f"[{cond}] skipped (missing LSTM or RNNG runs)")
            continue
        # seed-averaged per-pair accuracy in [0,1]
        l_pp = np.mean(lstm[cond], axis=0)
        r_pp = np.mean(rnng[cond], axis=0)
        gap_l, gap_r = gap_from_acc(l_pp, attr), gap_from_acc(r_pp, attr)
        dgap = gap_l - gap_r
        dacc = l_pp.mean() - r_pp.mean()

        boot_dgap = np.empty(args.bootstrap)
        boot_dacc = np.empty(args.bootstrap)
        for b in range(args.bootstrap):
            s = rng.integers(0, n, n)
            la, ra, am = l_pp[s], r_pp[s], attr[s]
            if am.all() or (~am).all():      # degenerate draw, no split
                boot_dgap[b] = np.nan
                boot_dacc[b] = la.mean() - ra.mean()
                continue
            boot_dgap[b] = gap_from_acc(la, am) - gap_from_acc(ra, am)
            boot_dacc[b] = la.mean() - ra.mean()
        bd = boot_dgap[~np.isnan(boot_dgap)]
        ci_lo, ci_hi = np.percentile(bd, [2.5, 97.5])
        p_dgap = float(np.mean(bd <= 0))          # H1: LSTM gap > RNNG gap (dgap > 0)
        acc_lo, acc_hi = np.percentile(boot_dacc, [2.5, 97.5])

        # seed-level (n=3) summary
        l_seed_gaps = [gap_from_acc(a.astype(float), attr) for a in lstm[cond]]
        r_seed_gaps = [gap_from_acc(a.astype(float), attr) for a in rnng[cond]]
        rows.append({
            "condition": cond,
            "gap_lstm": round(gap_l, 4), "gap_rnng": round(gap_r, 4),
            "delta_gap": round(dgap, 4),
            "delta_gap_ci_lo": round(ci_lo, 4), "delta_gap_ci_hi": round(ci_hi, 4),
            "delta_gap_p": round(p_dgap, 4),
            "delta_acc_all": round(dacc, 4),
            "delta_acc_ci_lo": round(acc_lo, 4), "delta_acc_ci_hi": round(acc_hi, 4),
            "gap_lstm_seedmean": round(float(np.mean(l_seed_gaps)), 4),
            "gap_lstm_seedstd": round(float(np.std(l_seed_gaps, ddof=1)) if len(l_seed_gaps) > 1 else 0.0, 4),
            "gap_rnng_seedmean": round(float(np.mean(r_seed_gaps)), 4),
            "gap_rnng_seedstd": round(float(np.std(r_seed_gaps, ddof=1)) if len(r_seed_gaps) > 1 else 0.0, 4),
            "n_lstm_seeds": len(lstm[cond]), "n_rnng_seeds": len(rnng[cond]),
            "n_pairs": n,
        })

    if not rows:
        raise SystemExit("No conditions had both LSTM and RNNG runs.")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'cond':<12} {'gap_LSTM':>9} {'gap_RNNG':>9} {'Δgap':>8} "
          f"{'95% CI':>16} {'p(Δgap<=0)':>11}")
    for r in rows:
        print(f"{r['condition']:<12} {r['gap_lstm']:>9.3f} {r['gap_rnng']:>9.3f} "
              f"{r['delta_gap']:>8.3f} "
              f"[{r['delta_gap_ci_lo']:>6.3f},{r['delta_gap_ci_hi']:>6.3f}] "
              f"{r['delta_gap_p']:>11.4f}")
    print(f"\nWrote {args.out_csv}")
    print("Δgap > 0 means the LSTM relies on the linear cue MORE than the RNNG "
          "(supports H3). p is the one-sided bootstrap mass at/below 0.")


if __name__ == "__main__":
    main()
