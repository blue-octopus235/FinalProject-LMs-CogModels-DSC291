# Results

**PRELIMINARY** — first local validation run, not the final-scale results.

- Run: `./run_sweep_local.sh` on an M1 Pro (MPS).
- Scale: LSTM, **50,000 training sentences/condition**, 5 epochs, **single seed (1)**.
- Eval: 19,819 held-out minimal pairs (1,528 with attractors), identical across conditions.

For the paper, rerun at full corpus size with seeds {1,2,3} on a T4
(`notebooks/colab_runner.ipynb`) and regenerate these figures.

| condition | rate | acc_all | acc_no_attr | acc_attr | gap |
|---|---|---|---|---|---|
| baseline | 0.0 | 0.861 | 0.877 | 0.662 | 0.216 |
| low | 0.004 | 0.861 | 0.879 | 0.651 | 0.229 |
| medium_low | 0.02 | 0.859 | 0.880 | 0.618 | 0.262 |
| medium_high | 0.10 | 0.827 | 0.848 | 0.570 | 0.278 |
| high | 0.50 | 0.657 | 0.670 | 0.500 | 0.170 |

Note: at 50% noise the attractor accuracy hits the 0.5 chance floor, so the
attractor *gap* shrinks not because hierarchy recovers but because both cells
collapse toward chance (floor effect) — interpret the gap alongside the raw
attractor accuracy, not on its own.
