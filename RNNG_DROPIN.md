# RNNG drop-in checklist (Sanjana)

Everything for the LSTM-vs-RNNG comparison is staged and ready. When Jackson's
preliminary RNNG run finishes, this is a ~5-minute task.

## What I need from Jackson
Push **`results/rnng_eval_results.csv`** — same columns as `results/eval_results.csv`
(`condition,rate,seed,acc_all,acc_no_attractor,acc_attractor,attractor_gap,mean_surprisal_correct_bits,...`),
one row per RNNG condition (baseline / medium_high / high). A `model` column is
optional (`plots_compare.py` adds it if missing).

## Steps once the CSV is in `results/`
1. Generate the comparison figures:
   ```bash
   python src/plots_compare.py        # writes results/compare_*.png
   ```
2. **Slide** — `presentation/sections/sec4_results.tex`: uncomment the
   "Preliminary: RNNG vs. LSTM" frame block at the bottom; fill the one
   bracketed `[TODO]` takeaway from the numbers.
3. **Report** — `report/sections/sec5_results.tex`: uncomment the
   "Preliminary: RNNG vs. LSTM" subsection; fill the three `XX.X` RNNG accuracy
   numbers (LSTM reference: baseline 0.961 / medium_high 0.934 / high 0.708).
4. **Discussion** — `report/sections/sec6_discussion.tex`: in the
   "Does structure buy robustness?" paragraph, replace the open *if-then* with
   the actual one-line finding (RNNG holds / does not hold the rule at high noise).
5. Rebuild:
   ```bash
   cd presentation && pdflatex main && pdflatex main
   cd ../report && pdflatex main && bibtex main && pdflatex main && pdflatex main
   ```

## Framing (do NOT overclaim)
RNNG = **preliminary**: 1 seed, 3 conditions, reduced epochs. Report a
**direction**, not a significance result. The full matched sweep (~58 GPU-hrs)
is paper-phase.
