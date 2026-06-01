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
  The local seed-1 trend is what we hoped for: the model does well on clean data
  (~0.96 overall) and steadily worse as we add noise (down to ~0.69 at the
  worst), and the attractor cases are consistently the tougher ones.

---
### Areas of ownership (from proposal)
- Data & training infra: Jackson (compute/coordination), Kelly (noise pipeline),
  Sanjana (training scripts/checkpointing)
- Eval & analysis: Sanjana (minimal-pair eval), Kelly (surprisal + rule probing),
  Jackson (visualization/stats)
- Writing & presentation: Kelly (intro/conclusion + coordination), Jackson
  (methods/setup), Sanjana (results/discussion + slides)
