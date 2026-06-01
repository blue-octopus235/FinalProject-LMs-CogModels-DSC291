# Robustness to Noise in Grammar Rule Learning

DSC 291 (Language Models as Cognitive Models, UCSD). Jackson Wilke, Kelly Fu, Sanjana Garimella.

We inject subject–verb agreement (SVA) errors into training data at 5 rates and ask how
noise affects whether a language model acquires the **hierarchical** agreement rule (agree with
the true subject) vs. the **linear** rule (agree with the nearest noun). Architectures:
**LSTM** (sequential baseline) vs. **RNNG** (syntactically-biased), per the professor's steer
toward a motivated syntactic-bias comparison (Wilcox et al. 2020; Sartran et al. 2022;
McCoy, Frankland & Linzen 2020).

## Data
Built by Kelly from the Linzen `agr_50_mostcommon_10K` corpus (≈1.58M sentences). Five corpora,
each `(orig_sentence, sentence, label)`, differing only in SVA error rate:

| condition | file | rate |
|---|---|---|
| baseline | `sva_corpus_baseline_rate_0.0.tsv` | 0.0 |
| low | `sva_corpus_low_rate_0.004.tsv` | 0.004 |
| medium_low | `sva_corpus_medium_low_rate_0.02.tsv` | 0.02 |
| medium_high | `sva_corpus_medium_high_rate_0.1.tsv` | 0.10 |
| high | `sva_corpus_high_rate_0.5.tsv` | 0.50 |

Corpora are **row-aligned** with `agr_50_mostcommon_10K.tsv` (verified: equal line counts, matching
rows, empirical error rates match the filenames). We train the LM on `orig_sentence` (real,
vocab-controlled words). Data files are git-ignored — share via Drive/Dropbox.

## Leak-free evaluation (important)
`src/data_utils.get_test_indices()` picks a **fixed, seeded set of 20k held-out row indices**.
Those rows are excluded from every training corpus, and the minimal-pair test set is built from
the **original** file at those same indices (`src/minimal_pairs.py`), reusing Kelly's exact
verb-flip (`src/verb_flip.py`) so eval and corruption stay consistent. The test set is identical
across all 5 conditions; only training noise varies. **Keep `--max_sentences` constant across
conditions** so training-set size doesn't confound the noise curve.

## Metrics
- **Minimal-pair accuracy**: does the model give higher P(verb | prefix) to the grammatical form?
- **Surprisal**: −log₂ P(correct verb).
- **Attractor gap** = acc(no attractor) − acc(attractor), where an attractor is an intervening
  noun of opposite number (`n_diff_intervening > 0`). A large/​widening gap = reliance on the
  linear (nearest-noun) rule.

## Run it
```bash
pip install -r requirements.txt
# train one condition (local smoke test uses a tiny subset on CPU)
python src/train.py    --condition baseline --max_sentences 3000 --epochs 1
python src/evaluate.py --checkpoint checkpoints/lstm_baseline_seed1.pt --condition baseline
python src/plots.py

# learning trajectory (RQ1: does noise delay or prevent acquisition?)
# save intermediate checkpoints during training, then score every step
python src/train.py      --condition baseline --max_sentences 3000 --epochs 2 --ckpt_every_steps 100
python src/trajectory.py --condition baseline --seed 1   # -> results/trajectory_results.csv
```
On a T4, use `notebooks/colab_runner.ipynb` (loops all 5 conditions). The full suite is
5 conditions × seeds; for the **Wednesday talk** run single-seed LSTM across all 5.

## RNNG
`notebooks/rnng_spike.ipynb` is the feasibility spike (the project's #1 risk). RNNG needs bracketed
parse trees and trains slower than the LSTM, so it is **paper-phase**: the talk stands on the 5
LSTM runs; RNNG-vs-LSTM is the headline comparison for the 6/11 paper. Fallback if RNNG won't
train in time: ON-LSTM (syntactic bias, no gold trees).

## Layout
```
src/   data_utils.py  lstm_lm.py  verb_flip.py  minimal_pairs.py  train.py  evaluate.py  plots.py
notebooks/  colab_runner.ipynb  rnng_spike.ipynb
data/  (git-ignored TSVs)   checkpoints/  results/
```
