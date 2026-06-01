# Linzen et al. (2016) — Assessing the ability of LSTMs to learn syntax-sensitive dependencies

**Full citation:** Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the ability of LSTMs to learn syntax-sensitive dependencies. *Transactions of the Association for Computational Linguistics*, 4, 521–535.

**PDF:** https://aclanthology.org/Q16-1037.pdf  
**ACL Anthology:** https://aclanthology.org/Q16-1037/

---

## What the paper does

Introduces the subject–verb agreement probing task as a benchmark for evaluating grammatical knowledge in RNNs. The authors take the trained language model and ask whether, for a sentence like *"The keys to the cabinet ___"*, it assigns higher probability to *are* than to *is*. By annotating how many **attractor nouns** (nouns of opposite number) intervene between subject and verb, the paper can separate models that learned the true hierarchical rule (agree with the syntactic subject) from those relying on a simpler linear heuristic (agree with the nearest noun).

The dataset — **`agr_50_mostcommon_10K`** (≈1.58M sentences extracted from Wikipedia with vocabulary capped at the 10K most frequent types) — is the exact corpus our project uses.

Key results:
- Vanilla LSTM LMs trained only on next-word prediction reach ~99% accuracy on simple subject–verb pairs.
- Accuracy degrades as attractor count increases, but is still well above chance, suggesting partial hierarchical sensitivity.
- Auxiliary supervision (training the LSTM to explicitly predict number) substantially improves performance, showing a data-efficiency tradeoff.

## Relevance to our project

- **Data source**: `agr_50_mostcommon_10K.tsv` *is* the Linzen dataset. Our data pipeline is built directly on top of it.
- **Evaluation paradigm**: Our minimal-pair accuracy metric, attractor gap, and the no-attractor/attractor partition all derive directly from this paper.
- **LSTM baseline**: Our sequential model replicates the Linzen et al. setup (2-layer word LSTM trained on next-word prediction), giving direct comparability to their published numbers.
- **Cite in intro/related work**: The opening framing of SVA as the canonical grammatical probe traces to this paper.

## Key terms

- **SVA (subject–verb agreement)**: the agreement relation our project studies
- **Attractor noun**: an intervening noun mismatching the subject in number; distinguishes hierarchical from linear strategies
- **Attractor gap**: accuracy(no attractor) − accuracy(attractor); our central metric

## Bibtex

```bibtex
@article{linzen2016assessing,
  title={Assessing the ability of {LSTMs} to learn syntax-sensitive dependencies},
  author={Linzen, Tal and Dupoux, Emmanuel and Goldberg, Yoav},
  journal={Transactions of the Association for Computational Linguistics},
  volume={4},
  pages={521--535},
  year={2016}
}
```
