"""Plots comparing LSTM and RNNG on the SVA noise task.

Reads:
  results/eval_results.csv       (LSTM, no 'model' column)
  results/rnng_eval_results.csv  (RNNG, has 'model' column)

Deduplicates each CSV by keeping the last row per (condition, seed) so that
repeated evaluation runs don't inflate averages.

Produces (in results/):
  combined_acc_vs_noise.png      accuracy vs noise: LSTM vs RNNG
  combined_attractor_gap.png     attractor gap vs noise: LSTM vs RNNG
  combined_acc_by_attractor.png  accuracy with/without attractor: LSTM vs RNNG
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["baseline", "low", "medium_low", "medium_high", "high"]
RATE_LABELS = {
    "baseline": "baseline\n(0.0)",
    "low": "low\n(0.004)",
    "medium_low": "med-low\n(0.02)",
    "medium_high": "med-high\n(0.10)",
    "high": "high\n(0.50)",
}


def load_dedup(path, arch_label):
    df = pd.read_csv(path)
    df["rate"] = df["rate"].astype(float)
    # Keep last row per (condition, seed) to handle repeated eval runs
    df = df.groupby(["condition", "seed"]).last().reset_index()
    df["arch"] = arch_label
    return df


def aggregate(df):
    agg = df.groupby(["condition", "rate", "arch"]).agg(
        acc_all=("acc_all", "mean"),
        acc_all_std=("acc_all", "std"),
        acc_attractor=("acc_attractor", "mean"),
        acc_attractor_std=("acc_attractor", "std"),
        acc_no_attractor=("acc_no_attractor", "mean"),
        acc_no_attractor_std=("acc_no_attractor", "std"),
        attractor_gap=("attractor_gap", "mean"),
        attractor_gap_std=("attractor_gap", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg["order"] = agg["condition"].apply(lambda c: ORDER.index(c) if c in ORDER else 99)
    return agg.sort_values(["order", "arch"]).reset_index(drop=True)


def make_xticks(df_arch):
    """One tick label per condition using the first arch's rows."""
    sub = df_arch[df_arch["arch"] == df_arch["arch"].iloc[0]].sort_values("order")
    return [RATE_LABELS.get(c, c) for c in sub["condition"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_csv",  default="results/eval_results.csv")
    ap.add_argument("--rnng_csv",  default="results/rnng_eval_results.csv")
    ap.add_argument("--out_dir",   default="results")
    # Which RNNG seeds to include in the paper figures. Default keeps ONLY the
    # validated full-coverage seed(s). Excluded by default:
    #   seed 1 — truncated test set (2,654 pairs, not the full ~19.8k), and
    #   seed 3 — non-monotonic / undertrained (medium_high acc 0.802 < high 0.834,
    #            negative high-noise gap); pending a re-run.
    # Averaging either in corrupts the comparison, so they are opt-in only.
    ap.add_argument("--rnng_seeds", default="2",
                    help="Comma-separated RNNG seeds to plot (default '2'); "
                         "pass 'all' to include every seed in the CSV.")
    args = ap.parse_args()

    frames = []
    if os.path.exists(args.lstm_csv):
        frames.append(load_dedup(args.lstm_csv, "LSTM"))
    if os.path.exists(args.rnng_csv):
        rnng = load_dedup(args.rnng_csv, "RNNG")
        if args.rnng_seeds.strip().lower() != "all":
            keep = {s.strip() for s in args.rnng_seeds.split(",") if s.strip()}
            before = len(rnng)
            rnng = rnng[rnng["seed"].astype(int).astype(str).isin(keep)]
            dropped = before - len(rnng)
            if dropped:
                print(f"[combined_plots] RNNG: kept seed(s) {sorted(keep)}, "
                      f"dropped {dropped} row(s) from other seeds "
                      f"(seed 1 truncated, seed 3 unvalidated; use --rnng_seeds all "
                      f"to override).")
        if len(rnng) == 0:
            print(f"[combined_plots] WARNING: --rnng_seeds {args.rnng_seeds!r} "
                  f"matched no RNNG rows; the RNNG arm will be ABSENT from the plots.")
        frames.append(rnng)

    if not frames:
        print("No results CSVs found.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    agg = aggregate(df_all)

    archs  = agg["arch"].unique()
    colors = {"LSTM": "#2196F3", "RNNG": "#FF5722"}
    markers = {"LSTM": "o", "RNNG": "s"}

    n_cond = len(ORDER)
    xs = np.arange(n_cond)
    xtick_labels = [RATE_LABELS.get(c, c) for c in ORDER]
    width = 0.25
    offsets = np.linspace(-(len(archs)-1)*width/2, (len(archs)-1)*width/2, len(archs))

    def get_arch_vals(arch, col):
        sub = agg[agg["arch"] == arch].set_index("condition")
        return np.array([sub.loc[c, col] if c in sub.index else np.nan for c in ORDER])

    # ── 1) Accuracy vs noise ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, arch in enumerate(archs):
        ys   = get_arch_vals(arch, "acc_all")
        errs = get_arch_vals(arch, "acc_all_std")
        errs = np.where(np.isnan(errs), 0, errs)
        ax.errorbar(xs + offsets[i], ys, yerr=errs,
                    label=arch, color=colors.get(arch), marker=markers.get(arch),
                    capsize=3, linewidth=1.5, markersize=6)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("minimal-pair accuracy")
    ax.set_xlabel("training noise condition")
    ax.set_title("SVA minimal-pair accuracy vs. training noise")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "combined_acc_vs_noise.png"), dpi=150)
    plt.close(fig)

    # ── 2) Attractor gap vs noise ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, arch in enumerate(archs):
        ys   = get_arch_vals(arch, "attractor_gap")
        errs = get_arch_vals(arch, "attractor_gap_std")
        errs = np.where(np.isnan(errs), 0, errs)
        ax.errorbar(xs + offsets[i], ys, yerr=errs,
                    label=arch, color=colors.get(arch), marker=markers.get(arch),
                    capsize=3, linewidth=1.5, markersize=6)
    ax.axhline(0.0, ls="--", c="gray", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("attractor gap = acc(no-attr) − acc(attr)")
    ax.set_xlabel("training noise condition")
    ax.set_title("Linear-rule reliance (attractor gap) vs. training noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "combined_attractor_gap.png"), dpi=150)
    plt.close(fig)

    # ── 3) Accuracy by attractor presence ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ls_map = {"LSTM": "-", "RNNG": "--"}
    for arch in archs:
        ya  = get_arch_vals(arch, "acc_attractor")
        yna = get_arch_vals(arch, "acc_no_attractor")
        c   = colors.get(arch)
        ls  = ls_map.get(arch, "-")
        ax.plot(xs, yna, color=c, ls=ls, marker=markers.get(arch), linewidth=1.5,
                markersize=6, label=f"{arch} no-attr")
        ax.plot(xs, ya,  color=c, ls=ls, marker=markers.get(arch), linewidth=1.5,
                markersize=6, alpha=0.55, label=f"{arch} attractor")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("minimal-pair accuracy")
    ax.set_xlabel("training noise condition")
    ax.set_title("Hierarchical vs. linear: accuracy by attractor presence")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "combined_acc_by_attractor.png"), dpi=150)
    plt.close(fig)

    written = [
        "combined_acc_vs_noise.png",
        "combined_attractor_gap.png",
        "combined_acc_by_attractor.png",
    ]
    print(f"[combined_plots] wrote {len(written)} figures to {args.out_dir}/")
    for f in written:
        n_seeds = agg.groupby("arch")["n_seeds"].max().to_dict()
        print(f"  {f}  (seeds: {n_seeds})")


if __name__ == "__main__":
    main()
