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

## RNNG (paper phase) — seed 2, full coverage

RNNG train+eval via `run_rnng_datahub.py` (150k sentences, 8 epochs). Raw rows:
`rnng_eval_results.csv`. Comparison figures: `combined_acc_vs_noise.png`,
`combined_attractor_gap.png`, `combined_acc_by_attractor.png` (from `src/combined_plots.py`,
run with `--rnng_csv` filtered to seed 2 so the truncated seed-1 rows are not averaged in).
**Seed 2 evaluated at full coverage (19,983 pairs); seeds 1 and 3 pending** (no error bars yet).

| condition | rate | acc_all | acc_no_attr | acc_attr | gap |
|---|---|---|---|---|---|
| baseline   | 0.0   | 0.979 | 0.981 | 0.960 | 0.020 |
| low        | 0.004 | 0.979 | 0.982 | 0.952 | 0.030 |
| medium_low | 0.02  | 0.978 | 0.979 | 0.958 | 0.022 |
| medium_high| 0.10  | 0.971 | 0.973 | 0.939 | 0.034 |
| high       | 0.50  | 0.834 | 0.837 | 0.800 | 0.036 |

Read: the RNNG attractor gap stays **flat and low** as noise rises (0.020 → 0.036) where the LSTM's
**grows** (0.073 → 0.198), and even at 50% noise the RNNG gap is below the LSTM's clean-data gap.
RNNG `acc_all` also stays above the LSTM at every rate (0.979 vs 0.961 baseline; 0.834 vs 0.708 at
high). This is the H3 "structure buys robustness" signal — pending seeds 1 and 3 for error bars.

> **Comparability — near-resolved at full coverage.** Seed 2 recovers the full held-out set
> (**19,983 pairs, 1,536 attractors**), within ~0.8% of the LSTM's **19,819 (1,528)**, so
> `acc_all` is now comparable across architectures up to that small coverage difference (the two
> sets are ~99% overlapping, not identical; the within-model attractor gap, a rate, is exact). An
> exact intersection-matched comparison (`src/match_subset.py` workflow) is left to future work.
> (Jackson's earlier seed-1 run scored only **2,654 pairs** — that build skipped the in-vocab
> filter and kept only beam-scorable pairs; those numbers are superseded by the full-coverage
> seed-2 run and must **not** be averaged with it. `rnng_eval_results.csv` retains both seeds, so
> filter to `seed==2` for the paper figures.) Still open: RNNG convergence check (val PPL at 8
> epochs) and a benepar tree-quality spot-check.
