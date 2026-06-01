# Results

Full-scale LSTM run: **150,000 training sentences/condition**, 5 epochs, **seeds {1,2,3}**.
- Run: `./run_sweep_multiseed.sh` on an M1 Pro (MPS).
- Eval: 19,819 held-out minimal pairs (1,528 with attractors), identical across conditions.
- Values below are **mean ± std over 3 seeds**. Raw per-seed rows: `eval_results.csv`.
- The earlier single-seed 50k validation is preserved in `eval_results_prelim_50k.csv`.

| condition | rate | acc_all | acc_no_attr | acc_attr | gap |
|---|---|---|---|---|---|
| baseline   | 0.0   | 0.957 ± 0.005 | 0.963 ± 0.003 | 0.876 ± 0.027 | 0.088 ± 0.024 |
| low        | 0.004 | 0.951 ± 0.008 | 0.959 ± 0.007 | 0.848 ± 0.025 | 0.111 ± 0.019 |
| medium_low | 0.02  | 0.942 ± 0.013 | 0.952 ± 0.009 | 0.828 ± 0.057 | 0.124 ± 0.048 |
| medium_high| 0.10  | 0.920 ± 0.018 | 0.935 ± 0.015 | 0.740 ± 0.048 | 0.195 ± 0.033 |
| high       | 0.50  | 0.698 ± 0.012 | 0.714 ± 0.014 | 0.506 ± 0.037 | 0.208 ± 0.046 |

## Reading the result

- **Overall accuracy degrades monotonically with noise** (0.957 → 0.698), with the sharp
  drop concentrated between 10% and 50% — small amounts of agreement noise are tolerated;
  large amounts break the model.
- **Attractor cases erode first and fastest.** Accuracy on pairs with an intervening
  attractor noun falls from 0.876 to chance (0.506) at 50% noise, while no-attractor
  accuracy stays high until the 50% condition.
- **The attractor gap grows monotonically** (0.088 → 0.208): as noise rises, the model
  leans harder on the linear (nearest-noun) heuristic over the hierarchical (true-subject)
  rule. Unlike the 50k single-seed prelim — where the gap *shrank* at 50% because both
  cells floored — at full scale the no-attractor cell hasn't floored (0.714), so the gap
  stays wide and the hierarchy-erosion story reads cleanly.

Figures (regenerated from the multiseed run, averaged over seeds):
`acc_vs_noise.png`, `attractor_gap.png`, `acc_by_attractor.png`.

## Next (paper phase, 6/11)

Add the RNNG arm (spike is GO — see `rnng/`) and compare its noise robustness and
attractor gap against the LSTM at matched scale.
