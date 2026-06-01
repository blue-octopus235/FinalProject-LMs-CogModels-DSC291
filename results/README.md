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

## Next (paper phase, 6/11)

Add the RNNG arm (spike is GO — see `rnng/`) and compare its noise robustness and
attractor gap against the LSTM at matched scale.
