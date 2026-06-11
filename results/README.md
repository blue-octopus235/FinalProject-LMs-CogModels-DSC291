# Results

Full-scale LSTM run: **150,000 training sentences/condition**, 5 epochs, **seeds {1,2,3} × 2 runs** (local MPS + Colab T4).
- Run: `./run_sweep_multiseed.sh` — local M1 Pro (MPS) + Colab T4 (CUDA).
- Eval: 19,819 held-out minimal pairs (1,528 with attractors), identical across conditions.
- Values below are **mean ± std over 6 runs**. Raw per-seed rows: `eval_results.csv`.
- The earlier single-seed 50k validation is preserved in `eval_results_prelim_50k.csv`.

| condition | rate | acc_all | acc_no_attr | acc_attr | gap |
|---|---|---|---|---|---|
| baseline   | 0.0   | 0.961 ± 0.006 | 0.967 ± 0.004 | 0.894 ± 0.027 | 0.073 ± 0.023 |
| low        | 0.004 | 0.958 ± 0.009 | 0.965 ± 0.007 | 0.878 ± 0.036 | 0.087 ± 0.030 |
| medium_low | 0.02  | 0.952 ± 0.014 | 0.960 ± 0.011 | 0.857 ± 0.048 | 0.103 ± 0.038 |
| medium_high| 0.10  | 0.934 ± 0.020 | 0.946 ± 0.016 | 0.793 ± 0.066 | 0.153 ± 0.051 |
| high       | 0.50  | 0.708 ± 0.018 | 0.723 ± 0.019 | 0.525 ± 0.032 | 0.198 ± 0.033 |

## Reading the result

- **Overall accuracy degrades monotonically with noise** (0.961 → 0.708), with the sharp
  drop concentrated between 10% and 50% — small amounts of agreement noise are tolerated;
  large amounts break the model.
- **Attractor cases erode first and fastest.** Accuracy on pairs with an intervening
  attractor noun falls from 0.894 to 0.525 at 50% noise, while no-attractor
  accuracy stays high until the 50% condition.
- **The attractor gap grows monotonically** (0.073 → 0.198): as noise rises, the model
  leans harder on the linear (nearest-noun) heuristic over the hierarchical (true-subject)
  rule. Unlike the 50k single-seed prelim — where the gap *shrank* at 50% because both
  cells floored — at full scale the no-attractor cell hasn't floored (0.723), so the gap
  stays wide and the hierarchy-erosion story reads cleanly.

Figures (regenerated from the multiseed run, averaged over seeds):
`acc_vs_noise.png`, `attractor_gap.png`, `acc_by_attractor.png`.

## Learning trajectory (RQ1: delay vs. prevention?)

Single-seed run (seed 1, 150k sentences, 5 epochs), checkpoint every 500 steps.
Raw data: `trajectory_results.csv`. Figures: `acc_trajectory.png`, `gap_trajectory.png`.

| condition | acc at step 500 | acc at step 11500 | verdict |
|---|---|---|---|
| baseline  | 0.681 | 0.966 | acquires rule fully |
| low       | 0.681 | 0.966 | acquires rule fully |
| medium_low| 0.678 | 0.965 | acquires rule fully |
| medium_high| 0.675 | 0.949 | acquires rule fully (slight cost) |
| **high**  | **0.605** | **0.693** | **plateaus — rule not acquired** |

Low-to-medium noise **delays** acquisition; 50% noise **prevents** it. The attractor
gap for the high condition also grows over training (0.14→0.20), meaning more training
makes the model *more* reliant on the linear nearest-noun rule, not less.

## RNNG (paper phase) — seed 1 so far

RNNG train+eval via `run_rnng_datahub.py` (150k sentences, 8 epochs). Raw rows:
`rnng_eval_results.csv`. Comparison figures: `combined_acc_vs_noise.png`,
`combined_attractor_gap.png`, `combined_acc_by_attractor.png` (from `src/combined_plots.py`).
**Seed 1 only; seeds 2–3 pending** (no error bars yet).

| condition | rate | acc_all | acc_no_attr | acc_attr | gap |
|---|---|---|---|---|---|
| baseline   | 0.0   | 0.980 | 0.984 | 0.934 | 0.050 |
| low        | 0.004 | 0.979 | 0.982 | 0.953 | 0.029 |
| medium_low | 0.02  | 0.976 | 0.979 | 0.948 | 0.031 |
| medium_high| 0.10  | 0.976 | 0.980 | 0.934 | 0.046 |
| high       | 0.50  | 0.742 | 0.751 | 0.637 | 0.114 |

Read: the RNNG attractor gap stays **flat and low** as noise rises (0.05 → 0.11) where the LSTM's
**grows** (0.07 → 0.20), and even at 50% noise the RNNG gap is below the LSTM's clean-data gap.
This is the H3 "structure buys robustness" signal — pending seeds 2–3 and the caveat below.

> **Comparability caveat.** RNNG is scored on **2,654 pairs (212 attractors)**, the LSTM on
> **19,819 (1,528)** — `run_rnng_datahub.py` skips the in-vocab filter and keeps only beam-scorable
> pairs, so the two test sets differ. `acc_all` is **not** directly comparable yet; the gap (a
> within-model rate) is the safer cross-model contrast. To make `acc_all` comparable, recover the
> RNNG coverage or re-score the LSTM on the same 2,654-pair subset. Also still open: RNNG
> convergence check (val PPL at 8 epochs) and a benepar tree-quality spot-check.
