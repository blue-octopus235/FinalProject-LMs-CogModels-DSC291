"""Plots from results/eval_results.csv (averages over seeds if present).

Produces:
  results/acc_vs_noise.png        minimal-pair accuracy vs noise rate
  results/attractor_gap.png       attractor gap (linear-rule reliance) vs noise
  results/acc_by_attractor.png    accuracy with vs without attractor

Add more architectures by giving each its own `arch` column / file; the plot
loops over whatever conditions are present.
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


def load(results_csv):
    df = pd.read_csv(results_csv)
    df["rate"] = df["rate"].astype(float)
    agg = df.groupby(["condition", "rate"]).agg(
        acc_all=("acc_all", "mean"),
        acc_all_std=("acc_all", "std"),
        acc_attractor=("acc_attractor", "mean"),
        acc_no_attractor=("acc_no_attractor", "mean"),
        attractor_gap=("attractor_gap", "mean"),
        surprisal=("mean_surprisal_correct_bits", "mean"),
    ).reset_index()
    agg["order"] = agg["condition"].apply(
        lambda c: ORDER.index(c) if c in ORDER else 99)
    return agg.sort_values("order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", default="results/eval_results.csv")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    df = load(args.results_csv)
    x = df["rate"].values
    xt = [f"{c}\n{r:g}" for c, r in zip(df["condition"], df["rate"])]

    # 1) accuracy vs noise
    plt.figure(figsize=(6, 4))
    yerr = df["acc_all_std"].fillna(0).values
    plt.errorbar(range(len(x)), df["acc_all"], yerr=yerr, marker="o", capsize=3)
    plt.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    plt.xticks(range(len(x)), xt)
    plt.ylabel("minimal-pair accuracy")
    plt.xlabel("training noise condition")
    plt.title("SVA accuracy vs. training noise (LSTM)")
    plt.ylim(0.4, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "acc_vs_noise.png"), dpi=150)

    # 2) attractor gap vs noise
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(x)), df["attractor_gap"], marker="s", color="firebrick")
    plt.xticks(range(len(x)), xt)
    plt.ylabel("attractor gap = acc(no-attr) - acc(attr)")
    plt.xlabel("training noise condition")
    plt.title("Reliance on linear (nearest-noun) cue vs. noise")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attractor_gap.png"), dpi=150)

    # 3) accuracy by attractor presence
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(x)), df["acc_no_attractor"], marker="o", label="no attractor")
    plt.plot(range(len(x)), df["acc_attractor"], marker="o", label="attractor")
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.xticks(range(len(x)), xt)
    plt.ylabel("minimal-pair accuracy")
    plt.xlabel("training noise condition")
    plt.title("Hierarchical vs. linear: accuracy by attractor presence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "acc_by_attractor.png"), dpi=150)
    print(f"[plots] wrote 3 figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
