# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DSC 291 final project (Jackson Wilke, Kelly Fu, Sanjana Garimella). Investigates whether LSTM vs. RNNG language models acquire the **hierarchical** SVA rule (agree with true subject) vs. the **linear** rule (agree with nearest noun) as training-data noise increases. Five SVA error rates: 0.0, 0.004, 0.02, 0.10, 0.50. Paper due 2026-06-11.

## Commands

```bash
pip install -r requirements.txt

# Train one condition (smoke test)
python src/train.py --condition baseline --max_sentences 3000 --epochs 1 --seed 1

# Evaluate a checkpoint
python src/evaluate.py --checkpoint checkpoints/lstm_baseline_seed1.pt --condition baseline --seed 1

# Generate result plots (reads results/eval_results.csv)
python src/plots.py

# Full local sweep (50k sentences, 1 seed)
bash run_sweep_local.sh

# Full multi-seed sweep (150k sentences, seeds 1-2-3 — use on Colab T4)
bash run_sweep_multiseed.sh

# RNNG train + eval, all conditions/seeds (run on datahub.ucsd.edu GPU)
python run_rnng_datahub.py --data_dir <path/to/data> --seeds 1 2 3

# LSTM-vs-RNNG comparison figures (reads both eval CSVs)
python src/combined_plots.py
```

Scripts must be run from the repo root (both shell scripts `cd` there automatically). `src/` modules import each other without a package prefix, so run python from the root as shown above.

Data files are git-ignored; share via Drive/Dropbox. Checkpoints are committed.

## Architecture

**Data flow**: `data_utils.py` → `train.py` → checkpoint → `evaluate.py` → `results/eval_results.csv` → `plots.py` → figures.

- **`src/data_utils.py`**: Vocabulary (`Vocab`), `LMDataset`, `get_test_indices()` (fixed seed-1234 20k held-out row indices shared across all conditions — do not change), `read_corpus_column()`, `CONDITIONS` dict mapping condition name → (filename, rate).
- **`src/lstm_lm.py`**: Two-layer LSTM LM (Gulordava 2018 style), weight tying when `emb_dim == hidden_dim`.
- **`src/train.py`**: Trains one condition/seed. Saves `checkpoints/lstm_{condition}_seed{N}.pt` with `model_state`, `args`, and `rate`. Vocab is built once from the original (clean) file and cached at `checkpoints/vocab.txt`.
- **`src/evaluate.py`**: Loads a checkpoint, builds minimal pairs, scores them, appends one row to `results/eval_results.csv`. Metrics: `acc_all`, `acc_attractor`, `acc_no_attractor`, `attractor_gap`, `mean_surprisal_correct_bits`.
- **`src/minimal_pairs.py`**: Builds the held-out test set from `agr_50_mostcommon_10K.tsv` at the fixed test indices. Each pair: prefix tokens, correct verb, incorrect verb (via `verb_flip.py`), and `has_attractor` flag.
- **`src/verb_flip.py`**: Flips singular↔plural verb forms using `lemminflect`. Must stay consistent between data corruption (training) and eval pair construction.
- **`src/plots.py`**: Aggregates `eval_results.csv` over seeds and writes three figures to `results/`.

## Leak-free design (critical invariant)

`get_test_indices()` returns the same 20k row indices for every condition. These rows are excluded from every training corpus. The minimal-pair test set is built from the **original** (unnoised) file at those same indices. Never change the seed or logic in `get_test_indices()` without regenerating all checkpoints. Always keep `--max_sentences` constant across conditions so training-set size doesn't confound the noise curve.

## RNNG (paper phase)

RNNG is being added for the 6/11 paper, not the talk. The full train+eval pipeline lives in
`run_rnng_datahub.py` (one script: parse → verb-swap per condition → `preprocess.py` → train →
`beam_search.py` surprisal eval → append to `results/rnng_eval_results.csv`). See
`rnng/RNNG_SPIKE_FINDINGS.md` for setup. Status (2026-06-08): **seed-1 results landed for all 5
conditions**; seeds 2–3 pending. Key points:
- Codebase: `aistairc/rnng-pytorch`. Apply `rnng/rnng_pytorch_fixes.patch` after cloning.
- Parse the **clean** corpus once with benepar (`spacy en_core_web_md` + `benepar_en3`); the tree structure is identical across all 5 noise conditions (only the verb terminal changes).
- On Mac: pass `--jobs 1` to `preprocess.py` (macOS lacks `os.sched_getaffinity`).
- Fallback if RNNG tree quality is too noisy: ON-LSTM (syntactic bias, no gold trees required).
- Surprisal eval: `beam_search.py --lm_output_file surprisals.txt` for per-word surprisals.

### Comparability: matched-subset workflow (do not skip)

`run_rnng_datahub.py` scores via `beam_search.py`, which only covers ~13% of the held-out set
(long sentences pruned) and skips the in-vocab verb filter — so RNNG's default `acc_all`
(**2,654 pairs**) is **not** on the same set as the LSTM's (**19,819**). To compare fairly, restrict
both models to the one set both can score. Every pair now carries a stable `idx` (source row),
and the pipeline emits the artifacts needed to align them:

```bash
# 1) RNNG run writes per-(cond,seed): results/rnng_covered_idx_*.json and
#    results/rnng_pairs_*.csv (per-pair idx/has_attractor/correct/surprisals)
python run_rnng_datahub.py --data_dir <data> --seeds 1 2 3

# 2) Build matched subset = (∩ RNNG covered) ∩ (LSTM in-vocab); recompute RNNG metrics on it
python src/match_subset.py
#    -> results/matched_idx.json, results/rnng_eval_results_matched.csv

# 3) Re-score each LSTM checkpoint on the SAME subset (cheap, no retraining)
python src/evaluate.py --checkpoint checkpoints/lstm_<cond>_seed<N>.pt --condition <cond> \
    --seed <N> --restrict_indices results/matched_idx.json \
    --results_csv results/eval_results_matched.csv
#    (src/match_subset.py prints a ready-to-run loop over all checkpoints)

# 4) Plot the matched comparison
python src/combined_plots.py --lstm_csv results/eval_results_matched.csv \
    --rnng_csv results/rnng_eval_results_matched.csv
```

Until step 4, the within-model attractor gap (a rate, not a count) is the safer cross-model
contrast. `src/combined_plots.py` also deduplicates repeated rows.

### RNNG sanity outputs

`run_rnng_datahub.py` also writes: `results/rnng_train_ppl.csv` (validation PPL/WordPPL per
validation event, scraped from the training log — check 8 epochs actually converged) and
`results/rnng_parse_quality.json` + `results/rnng_tree_sample.txt` (benepar parse failure rate +
a 40-tree hand-check sample for the §7 tree-quality validity risk on lowercased/anonymized tokens).
The full training output is teed to `<work_dir>/logs/train_<cond>_seed<N>.log` — that log is the
authoritative record; the PPL CSV is a best-effort parse of rnng-pytorch's `PPL: … WordPPL: …`
validation lines.
