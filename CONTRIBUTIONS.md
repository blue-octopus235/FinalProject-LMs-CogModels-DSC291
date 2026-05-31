# Contributions

Running log of who did what, kept current so the final report's contributions
statement is a copy-paste. Add a dated bullet when you finish something
meaningful (code, analysis, writing, coordination — not just commits). Keep it
short.

## Jackson Wilke
- **2026-05-31** — Stood up the end-to-end LSTM pipeline: data loading + shared
  vocab, leak-free 20k-row held-out split, minimal-pair eval (accuracy,
  surprisal, attractor gap), training script, plotting. Verified corpus
  alignment + noise rates. Added MPS/CUDA support and the Colab runner +
  RNNG-spike notebooks. Drafted the methods section. Ran the first local
  5-condition baseline sweep. (`src/`, `notebooks/`, `README.md`,
  `ProjectProposal/methods_draft.md`)

## Kelly Fu
- **(pre-2026-05-31)** — Built the noise-injection pipeline and the 5 corrupted
  corpora (`flip_verb_agreement_perfect`, `inject_sva_errors`); validated 5,000
  sampled sentences for correct injection. (`data/Untitled.ipynb`, the 5
  `sva_corpus_*` files)

## Sanjana Garimella
- _(add entries as work lands)_

---
### Areas of ownership (from proposal)
- Data & training infra: Jackson (compute/coordination), Kelly (noise pipeline),
  Sanjana (training scripts/checkpointing)
- Eval & analysis: Sanjana (minimal-pair eval), Kelly (surprisal + rule probing),
  Jackson (visualization/stats)
- Writing & presentation: Kelly (intro/conclusion + coordination), Jackson
  (methods/setup), Sanjana (results/discussion + slides)
