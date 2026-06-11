"""Build the matched LSTM-vs-RNNG comparison subset and recompute RNNG metrics on it.

Why: the RNNG pipeline scores pairs via beam_search, which only covers ~13% of the
held-out set (long sentences get pruned), and it skips the in-vocab verb filter the
LSTM uses. So the two models' default `acc_all` are on different pair sets and are not
directly comparable. This script defines the one set BOTH models can score on:

    matched_idx = (pairs covered by EVERY RNNG run) ∩ (LSTM in-vocab pairs)

and recomputes the RNNG metrics restricted to it. The LSTM side is produced by running
evaluate.py with --restrict_indices results/matched_idx.json (printed at the end).

Inputs  : results/rnng_pairs_{cond}_seed{seed}.csv  (written by run_rnng_datahub.py)
          checkpoints/vocab.txt, data/<ORIGINAL_FILE>
Outputs : results/matched_idx.json
          results/rnng_eval_results_matched.csv   (RNNG metrics on the matched subset)

Run from the repo root:  python src/match_subset.py
"""
import argparse
import csv
import glob
import json
import math
import os
import re

from data_utils import CONDITIONS, ORIGINAL_FILE, get_test_indices, Vocab
from minimal_pairs import build_minimal_pairs

PAIRS_GLOB = "results/rnng_pairs_*.csv"
FNAME_RE = re.compile(r"rnng_pairs_(?P<cond>.+)_seed(?P<seed>\d+)\.csv$")


def load_rnng_pairs(paths):
    """Return {(cond, seed): {idx: {has_attractor, correct}}}."""
    runs = {}
    for path in paths:
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f"  skip (unparseable name): {path}")
            continue
        cond, seed = m["cond"], int(m["seed"])
        rows = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[int(r["idx"])] = {
                    "has_attractor": bool(int(r["has_attractor"])),
                    "correct": bool(int(r["correct"])),
                }
        runs[(cond, seed)] = rows
        print(f"  loaded {len(rows):>6} pairs  {cond} seed{seed}")
    return runs


def metrics(records):
    """records: list of {has_attractor, correct}. Returns the standard metric dict."""
    def acc(items):
        if not items:
            return float("nan"), 0
        return sum(1 for r in items if r["correct"]) / len(items), len(items)

    attr = [r for r in records if r["has_attractor"]]
    noattr = [r for r in records if not r["has_attractor"]]
    acc_all, n_all = acc(records)
    acc_attr, n_attr = acc(attr)
    acc_noattr, n_noattr = acc(noattr)
    gap = (acc_noattr - acc_attr) if (n_attr and n_noattr) else float("nan")
    return {
        "acc_all": acc_all, "n_all": n_all,
        "acc_no_attractor": acc_noattr, "n_no_attractor": n_noattr,
        "acc_attractor": acc_attr, "n_attractor": n_attr,
        "attractor_gap": gap,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab", default="checkpoints/vocab.txt")
    ap.add_argument("--pairs_glob", default=PAIRS_GLOB)
    ap.add_argument("--out_idx", default="results/matched_idx.json")
    ap.add_argument("--out_csv", default="results/rnng_eval_results_matched.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pairs_glob))
    if not paths:
        raise SystemExit(f"No RNNG per-pair files at {args.pairs_glob} — run "
                         f"run_rnng_datahub.py (it writes rnng_pairs_*.csv) first.")
    print(f"RNNG per-pair files ({len(paths)}):")
    runs = load_rnng_pairs(paths)

    # Warn if the run set looks incomplete: a re-run with only --seeds 2 3 would
    # build the matched set from those seeds while a stale seed-1 row still sits in
    # rnng_eval_results.csv. Flag any (cond, seed) missing from conds×seeds present.
    conds = sorted({c for c, _ in runs})
    seeds = sorted({s for _, s in runs})
    missing = [(c, s) for c in conds for s in seeds if (c, s) not in runs]
    expected = set(CONDITIONS)
    if missing:
        print(f"  WARNING: missing per-pair files for {missing} "
              f"(found conds={conds}, seeds={seeds}). The matched subset will be "
              f"built only from the files present.")
    if set(conds) != expected:
        print(f"  WARNING: conditions present {conds} != all 5 {sorted(expected)}.")

    # pairs covered by EVERY RNNG run (so every run shares one denominator)
    rnng_covered = set.intersection(*(set(r) for r in runs.values()))
    print(f"\nRNNG covered (intersection over {len(runs)} runs): {len(rnng_covered)}")

    # LSTM in-vocab pairs
    vocab = Vocab.load(args.vocab)
    lstm_pairs, _ = build_minimal_pairs(
        os.path.join(args.data_dir, ORIGINAL_FILE), get_test_indices(), vocab=vocab)
    lstm_idx = {p["idx"] for p in lstm_pairs}
    print(f"LSTM in-vocab pairs: {len(lstm_idx)}")

    matched = sorted(rnng_covered & lstm_idx)
    print(f"\n==> matched subset (both models can score): {len(matched)}")
    if not matched:
        raise SystemExit("Empty matched set — check inputs.")

    os.makedirs(os.path.dirname(args.out_idx), exist_ok=True)
    with open(args.out_idx, "w") as f:
        json.dump(matched, f)
    print(f"Wrote {args.out_idx}")

    # Recompute RNNG metrics on the matched subset
    matched_set = set(matched)
    cols = ["model", "condition", "rate", "seed", "acc_all", "acc_no_attractor",
            "acc_attractor", "attractor_gap", "n_all", "n_attractor",
            "n_no_attractor", "subset"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (cond, seed), rows in sorted(runs.items()):
            recs = [rows[i] for i in matched if i in rows]
            res = metrics(recs)
            res.update({"model": "rnng", "condition": cond,
                        "rate": CONDITIONS[cond][1], "seed": seed,
                        "subset": "matched"})
            w.writerow({k: res.get(k) for k in cols})
            print(f"  {cond:<12} seed{seed}: acc_all={res['acc_all']:.3f} "
                  f"gap={res['attractor_gap']:.3f} (n={res['n_all']})")
    print(f"Wrote {args.out_csv}")

    print("\nNext — score the LSTM on the SAME subset, e.g.:")
    print("  for ck in checkpoints/lstm_*.pt; do")
    print('    cond=$(basename "$ck" | sed -E "s/lstm_(.*)_seed[0-9]+\\.pt/\\1/")')
    print('    seed=$(basename "$ck" | sed -E "s/.*_seed([0-9]+)\\.pt/\\1/")')
    print('    python src/evaluate.py --checkpoint "$ck" --condition "$cond" '
          '--seed "$seed" \\')
    print(f"      --restrict_indices {args.out_idx} "
          "--results_csv results/eval_results_matched.csv")
    print("  done")
    print("\nThen plot the matched comparison:")
    print("  python src/combined_plots.py \\")
    print("    --lstm_csv results/eval_results_matched.csv \\")
    print(f"    --rnng_csv {args.out_csv}")


if __name__ == "__main__":
    main()
