# Kuncoro et al. (2018) — LSTMs Can Learn Syntax-Sensitive Dependencies Well, But Modeling Structure Makes Them Better

**Full citation:** Kuncoro, A., Dyer, C., Hale, J., Yogatama, D., Clark, S., & Blunsom, P. (2018). LSTMs can learn syntax-sensitive dependencies well, but modeling structure makes them better. In *Proceedings of ACL 2018*, 1426–1436.

**PDF:** https://aclanthology.org/P18-1132.pdf  
**ACL Anthology:** https://aclanthology.org/P18-1132/

---

## What the paper does

Directly compares a sequential LSTM language model against the **RNNG** (Recurrent Neural Network Grammar) on the Linzen subject–verb agreement task, asking whether an explicit hierarchical/structural bias improves syntactic generalization. Findings:
- A well-tuned LSTM is already a strong agreement learner — better than earlier work suggested.
- The **RNNG outperforms the LSTM**, and the gap widens as the number of attractors grows, i.e. precisely in the hardest hierarchical cases.
- Ablations show the benefit comes from composing phrases into structured representations rather than from extra parameters.

## Relevance to our project

- **This is the empirical precedent for our central architectural hypothesis.** Our project asks whether RNNG's structural bias buys *robustness to noise*; Kuncoro et al. establish that, on *clean* data, structure already buys robustness to *attractors*. Our noise sweep is the natural extension — does that structural advantage also hold up when the training signal itself is corrupted?
- **Motivates the RNNG-vs-LSTM design** (alongside the professor's recommended Wilcox 2020 / Sartran 2022). It is the cleanest head-to-head on our exact task and metric.
- **Predicts the shape of our cross-architecture result**: if their attractor-gap advantage for RNNG persists under noise, we expect RNNG to hold the hierarchical rule at noise levels where the LSTM has already defaulted to the linear cue.
- **Already in `refs.bib`** (`kuncoro2018`): cited in methods/discussion; this summary backs that citation.

## Key terms

- **RNNG (Recurrent Neural Network Grammar)**: a model that jointly generates symbols and the tree structure over them, giving an explicit hierarchical bias
- **Composition function**: the operation that builds a phrase representation from its children — identified as the source of RNNG's edge
- **Structural bias**: an architectural inductive bias toward tree-structured (hierarchical) generalization

## Bibtex

```bibtex
@inproceedings{kuncoro2018lstms,
  title={{LSTMs} Can Learn Syntax-Sensitive Dependencies Well, But Modeling Structure Makes Them Better},
  author={Kuncoro, Adhiguna and Dyer, Chris and Hale, John and Yogatama, Dani and Clark, Stephen and Blunsom, Phil},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  pages={1426--1436},
  year={2018}
}
```
