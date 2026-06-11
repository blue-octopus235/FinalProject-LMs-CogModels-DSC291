# Marvin & Linzen (2018) — Targeted Syntactic Evaluation of Language Models

**Full citation:** Marvin, R., & Linzen, T. (2018). Targeted syntactic evaluation of language models. In *Proceedings of EMNLP 2018*, 1192–1202.

**PDF:** https://aclanthology.org/D18-1151.pdf  
**ACL Anthology:** https://aclanthology.org/D18-1151/

---

## What the paper does

Generalizes the Linzen et al. (2016) agreement probe into a broad **minimal-pair benchmark**. Instead of measuring per-token agreement accuracy, the authors construct matched grammatical/ungrammatical sentence pairs that differ in exactly one structurally relevant feature, and score a model as correct when it assigns higher probability to the grammatical member. The construction is templatic, which lets them control the intervening material precisely.

Coverage goes well beyond simple subject–verb agreement to harder structural environments — agreement across a relative clause, across a prepositional phrase, in sentential complements, plus reflexive anaphora and negative polarity items. Key results:
- LSTM LMs handle simple agreement well but degrade sharply in environments with embedded clauses and multiple attractors.
- A model supervised explicitly for syntax outperforms the plain LM in the hardest conditions, again showing the structural-evidence gap.

## Relevance to our project

- **Evaluation paradigm**: This is the paper that formalizes the *minimal-pair, higher-probability-to-the-grammatical-form* metric we use directly (BLiMP later scales it up). It is the methodological bridge between Linzen's per-token probe and our minimal-pair accuracy.
- **Attractor environments**: Their finding that accuracy collapses specifically when attractors and embeddings are present mirrors our attractor-gap result — the conditions where the hierarchical and linear rules diverge are exactly where the model is fragile.
- **Already in `refs.bib`** (`marvin2018`): cited in methods/related work; this summary backs that citation.

## Key terms

- **Minimal pair**: matched grammatical/ungrammatical sentences differing in one feature; scored by relative probability
- **Targeted syntactic evaluation**: controlled templatic test sets isolating one syntactic phenomenon at a time
- **Agreement attraction environment**: configurations (relative clauses, PPs) where an intervening noun can lure the model toward the linear rule

## Bibtex

```bibtex
@inproceedings{marvin2018targeted,
  title={Targeted Syntactic Evaluation of Language Models},
  author={Marvin, Rebecca and Linzen, Tal},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={1192--1202},
  year={2018}
}
```
