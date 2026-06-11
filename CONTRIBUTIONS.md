# Contributions

Running log of who did what, kept current so the final report's contributions
statement is a copy-paste. Add a dated bullet when you finish something
meaningful (code, analysis, writing, coordination — not just commits). Keep it
short.

## Jackson Wilke
- **2026-05-07** — Wrote the first version of the project proposal.
- **2026-05-31** — Stood up the end-to-end LSTM pipeline: data loading + shared
  vocab, leak-free 20k-row held-out split, minimal-pair eval (accuracy,
  surprisal, attractor gap), training script, plotting. Verified corpus
  alignment + noise rates. Added MPS/CUDA support and the Colab runner +
  RNNG-spike notebooks. Drafted the methods section. Ran the first local
  5-condition baseline sweep. (`src/`, `notebooks/`, `README.md`,
  `ProjectProposal/methods_draft.md`)
- **2026-05-31** — Ran the RNNG feasibility spike (the project's #1 risk) to a
  **GO**: got `aistairc/rnng-pytorch` installing + training end-to-end, and
  confirmed the full pipeline on our own data — real Linzen sentences →
  benepar parses → `preprocess.py` → RNNG trains (val PPL drops). Found/patched
  a NumPy-2 crash that also hits Colab. Documented the path, fixes, and a key
  efficiency insight (parse once, reuse trees across all noise conditions) plus
  the parser-quality validity risk for the paper. (`rnng/`)
- **2026-05-31** — Upgraded LSTM results from the 50k single-seed prelim to the
  full 150k × seeds {1,2,3} sweep. Key finding: overall accuracy degrades
  monotonically with SVA noise (0.957 → 0.698) and the attractor gap grows
  monotonically (0.088 → 0.208), consistent with a hierarchical-rule account.
  Preserved prelim in `results/eval_results_prelim_50k.csv`; refreshed
  `results/README.md` with mean ± std table and updated all three figures.
  (`results/`, `run_sweep_multiseed.sh`)
- **2026-05-31** — Reorganized `literature/` into `pdfs/` and `summaries/`
  subdirectories; wrote a structured one-page summary for each of the 12 core
  papers (Linzen 2016, Gulordava 2018, Dyer 2016, McCoy 2020, Warstadt 2020 +
  2023, Sartran 2022, Shen 2019, Gauthier 2020, Wilcox 2020, Chomsky 1980).
  Added `literature/README.md` index table and authored `CLAUDE.md` with
  commands, data-flow architecture, and the leak-free invariant. (`literature/`,
  `CLAUDE.md`)

- **2026-06-01** — Scaffolded the final paper in `report/`: ACL-style LaTeX
  template (official `acl.sty`/`acl_natbib.bst`, ACL Limitations section),
  one-file-per-section skeleton (abstract, intro, background, methods,
  experiments, results, discussion, conclusion, limitations, appendix) with
  per-section owner + TODO stubs assigned from the proposal's writing split.
  Wired figures to reference `results/` in place via `\graphicspath`, seeded
  `refs.bib` with the core SVA/RNNG citations, added a `latexmk` `Makefile`
  (`make` / `watch` / `clean` / `purge`) and a `report/.gitignore` for LaTeX
  aux files. Verified the skeleton compiles. (`report/`)

- **2026-06-01** — Scaffolded the talk in `presentation/`: Beamer deck
  (Madrid theme, 16:9) mirroring the report's framework — one-file-per-section
  skeleton (motivation, background, methods, results, discussion) with frame
  stubs, per-frame owner + TODO comments from the proposal's split, results
  frames pre-wired to `results/` figures via `\graphicspath`, plus a `latexmk`
  `Makefile` and `presentation/.gitignore`. Verified it compiles. (`presentation/`)

- **2026-06-08** — Built the end-to-end RNNG train+eval pipeline for datahub.ucsd.edu
  (`run_rnng_datahub.py`): parse the clean corpus once, swap the verb terminal per
  noise condition, `preprocess.py`, train per condition/seed, then score the shared
  minimal pairs via `beam_search.py` surprisals into `results/rnng_eval_results.csv`.
  Added inline rnng-pytorch fixes (PyTorch-2.6 `weights_only`, NumPy-2.x) and live
  output filtering. Ran **seed 1 across all 5 conditions** and wrote
  `src/combined_plots.py` for the LSTM-vs-RNNG comparison figures
  (`results/combined_*.png`). Headline: the RNNG attractor gap stays flat/low as noise
  rises (0.05→0.11) where the LSTM's grows (0.07→0.20) — the H3 "structure buys
  robustness" signal, pending seeds 2–3. Flagged the test-set comparability caveat
  (RNNG scored on 2,654 pairs vs. the LSTM's 19,819) in the docs. (`run_rnng_datahub.py`,
  `src/combined_plots.py`, `results/`)

- **2026-06-01** — Wrote the methods slides: filled the talk's methods section,
  expanding the stubs into five focused frames (noise injection + five corrupted
  corpora, leak-free 20k held-out protocol, the minimal-pair test, LSTM-vs-RNNG
  models, and metrics/rule-probing) with key bullets and equations
  (verb-flip rates, the $P(v^{+}) > P(v^{-})$ criterion, surprisal, attractor
  gap). Verified the deck compiles. (`presentation/sections/sec3_methods.tex`)

## Kelly Fu
- **2026-05-27** — Updated the project proposal with the professor's feedback
  (switched the primary model GPT-2 → RNNG as a syntactically-biased comparison,
  added the data-acquisition plan; background-papers section still pending as of
  5/31).
- **2026-05-29** — Built the noise-injection pipeline and the 5 corrupted
  corpora (`flip_verb_agreement_perfect`, `inject_sva_errors`); validated 5,000
  sampled sentences for correct injection. (`data/Untitled.ipynb`, the 5
  `sva_corpus_*` files)

## Sanjana Garimella
- **2026-05-11** — Went through the proposal draft and cleaned up the research
  questions and the setup section.
- **2026-05-29** — Talked to the prof to get a rough sense of the dataset and
  its demographics, and grabbed the data from the Linzen et al. paper.
- **2026-06-01** — Set up checkpointing during training so we can check how the
  model does partway through, not just at the end. Added `--ckpt_every_steps`
  to `train.py` to save those checkpoints, plus `src/trajectory.py` to run the
  minimal-pair eval on each one.
- **2026-06-01** — Ran the evals across all five noise levels and looked at how
  the models did — overall accuracy, plus accuracy with and without an attractor.
  The trend is what we hoped for: the model does well on clean data and steadily
  worse as we add noise, and the attractor cases are consistently the tougher ones.
- **2026-06-01** — Ran the full 3-seed sweep on Colab (T4, 150k sentences each)
  and pulled the results together with the earlier sweep — 6 runs per condition
  in total. The noise effect is really clear: accuracy drops from ~0.96 at
  baseline down to ~0.71 at the highest noise rate, and the attractor gap nearly
  triples, meaning the model leans more on the nearest-noun shortcut the noisier
  the training data gets. Updated `results/eval_results.csv` and the figures.
- **2026-06-01** — Ran the learning-trajectory sweep on Colab and added the
  results to `results/trajectory_results.csv`. The curves answer RQ1 directly:
  low-noise conditions all climb to ~0.96 by the end of training (noise delays
  but doesn't stop acquisition), while the 50% noise condition plateaus around
  0.69 and never gets there — that's prevention. Added trajectory plots to
  `src/plots.py` (`acc_trajectory.png`, `gap_trajectory.png`).
- **2026-06-02** — Wrote the Results and Discussion sections (report + slides)
  from the final numbers: the threshold-like accuracy drop, the widening
  attractor gap, and the delay-vs-prevention trajectory story. Expanded the
  `literature/` review with four papers and summaries (Marvin & Linzen 2018,
  Kuncoro 2018, Lakretz 2019, Evanson 2023) and added the new references to
  `report/refs.bib`. (`report/sections/sec5_results.tex`, `sec6_discussion.tex`,
  `presentation/sections/sec4_results.tex`, `sec5_discussion.tex`, `literature/`)
### Areas of ownership (from proposal)
- Data & training infra: Jackson (compute/coordination), Kelly (noise pipeline),
  Sanjana (training scripts/checkpointing)
- Eval & analysis: Sanjana (minimal-pair eval), Kelly (surprisal + rule probing),
  Jackson (visualization/stats)
- Writing & presentation: Kelly (intro/conclusion + coordination), Jackson
  (methods/setup), Sanjana (results/discussion). Slides split by topic to mirror
  report sections — Kelly (motivation/background), Jackson (methods), Sanjana
  (results/discussion); Sanjana coordinates/assembles the deck.
