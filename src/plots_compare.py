"""Comparison plots: LSTM vs RNNG across noise conditions.

Reads results/eval_results.csv (LSTM) and results/rnng_eval_results.csv (RNNG),
merges them, and produces side-by-side comparison figures.

Produces:
  results/compare_acc_vs_noise.png      accuracy vs noise, both models
  results/compare_attractor_gap.png     attractor gap vs noise, both models
  results/compare_acc_by_attractor.png  attractor split, both models
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["baseline", "low", "medium_low", "medium_high", "high"]
COLORS = {"lstm": "#1f77b4", "rnng": "#d62728"}
MARKERS = {"lstm": "o", "rnng": "s"}
LABELS = {"lstm": "LSTM", "rnng": "RNNG"}


def load_csv(path, model_name):
    df = pd.read_csv(path)
    if "model" not in df.columns:
        df["model"] = model_name
    df["rate"] = df["rate"].astype(float)
    return df


def aggregate(df):
    agg = df.groupby(["model", "condition", "rate"]).agg(
        acc_all=("acc_all", "mean"),
        acc_all_std=("acc_all", "std"),
        acc_attractor=("acc_attractor", "mean"),
        acc_no_attractor=("acc_no_attractor", "mean"),
        attractor_gap=("attractor_gap", "mean"),
        surprisal=("mean_surprisal_correct_bits", "mean"),
    ).reset_index()
    agg["order"] = agg["condition"].apply(
        lambda c: ORDER.index(c) if c in ORDER else 99)
    return agg.sort_values(["model", "order"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_csv",  default="results/eval_results.csv")
    ap.add_argument("--rnng_csv",  default="results/rnng_eval_results.csv")
    ap.add_argument("--out_dir",   default="results")
    args = ap.parse_args()

    frames = []
    for path, name in [(args.lstm_csv, "lstm"), (args.rnng_csv, "rnng")]:
        if os.path.exists(path):
            frames.append(load_csv(path, name))
        else:
            print(f"[compare] {path} not found, skipping {name}")
    if not frames:
        print("[compare] no data found, exiting")
        return
    df = pd.concat(frames, ignore_index=True)
    agg = aggregate(df)

    models = sorted(agg["model"].unique())
    xt_ref = None

    def get_model_data(model):
        sub = agg[agg["model"] == model].copy()
        xs = list(range(len(sub)))
        xt = [f"{c}\n{r:g}" for c, r in zip(sub["condition"], sub["rate"])]
        return sub, xs, xt

    # --- 1) accuracy vs noise ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        sub, xs, xt = get_model_data(m)
        yerr = sub["acc_all_std"].fillna(0).values
        ax.errorbar(xs, sub["acc_all"], yerr=yerr,
                    marker=MARKERS[m], color=COLORS[m], label=LABELS[m], capsize=3)
        if xt_ref is None:
            xt_ref = (xs, xt)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(xt_ref[0]); ax.set_xticklabels(xt_ref[1])
    ax.set_ylabel("minimal-pair accuracy")
    ax.set_xlabel("training noise condition")
    ax.set_title("SVA accuracy vs. training noise: LSTM vs. RNNG")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "compare_acc_vs_noise.png"), dpi=150)
    plt.close(fig)

    # --- 2) attractor gap vs noise ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        sub, xs, xt = get_model_data(m)
        ax.plot(xs, sub["attractor_gap"],
                marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xticks(xt_ref[0]); ax.set_xticklabels(xt_ref[1])
    ax.set_ylabel("attractor gap = acc(no-attr) − acc(attr)")
    ax.set_xlabel("training noise condition")
    ax.set_title("Linear-rule reliance (attractor gap) vs. noise: LSTM vs. RNNG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "compare_attractor_gap.png"), dpi=150)
    plt.close(fig)

    # --- 3) attractor split ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        sub, xs, xt = get_model_data(m)
        ax.plot(xs, sub["acc_no_attractor"],
                marker=MARKERS[m], color=COLORS[m], linestyle="-",
                label=f"{LABELS[m]} no-attractor")
        ax.plot(xs, sub["acc_attractor"],
                marker=MARKERS[m], color=COLORS[m], linestyle="--",
                label=f"{LABELS[m]} attractor")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(xt_ref[0]); ax.set_xticklabels(xt_ref[1])
    ax.set_ylabel("minimal-pair accuracy")
    ax.set_xlabel("training noise condition")
    ax.set_title("Hierarchical vs. linear rule by attractor presence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "compare_acc_by_attractor.png"), dpi=150)
    plt.close(fig)

    print(f"[compare] wrote 3 figures to {args.out_dir}/")

    # Print summary table
    print("\n=== Summary ===")
    print(agg[["model", "condition", "rate", "acc_all", "attractor_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
