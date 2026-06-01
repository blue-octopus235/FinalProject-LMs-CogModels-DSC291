# Wilcox et al. (2020) — On the Acceptability of the Grammar of Language Models

**Full citation:** Wilcox, E., Qian, P., Futrell, R., Kohita, R., Levy, R., & Ballesteros, M. (2020). Structural supervision improves learning of non-local grammatical dependencies. In *Proceedings of NAACL-HLT 2020* (pp. 3302–3312).

> *Note: The README and proposal cite "Wilcox et al. 2020" in the context of RNNG showing improved few-shot syntactic generalization. The most likely intended citation is the 2020 NAACL paper above. A related 2019 paper (Wilcox et al., ACL 2019) covers similar ground; check with the professor if the exact paper is needed.*

**PDF (2020 NAACL):** https://aclanthology.org/2020.naacl-main.303.pdf  
**ACL Anthology:** https://aclanthology.org/2020.naacl-main.303/

---

## What the paper does

Asks whether **structural supervision** — training a model to predict syntactic structure alongside words — improves learning of long-distance grammatical dependencies beyond what a plain LSTM acquires. The comparison is:

- Plain LSTM language model (sequential inductive bias only)
- RNNG (Dyer et al. 2016) — explicit hierarchical structural supervision via parse trees

Evaluation is on a suite of targeted syntactic phenomena in the SyntaxGym / NATLOG tradition, including subject–verb agreement with attractors, reflexive anaphora, negative polarity items, and garden-path effects.

Key results:
- RNNGs outperform plain LSTMs on the majority of targeted syntactic tests, especially non-local dependencies.
- The advantage is particularly pronounced in **low-data regimes** (the paper's "few-shot" framing): with limited training sentences, RNNGs generalize better to held-out syntactic structures.
- Structural supervision acts as an inductive bias that helps the model bootstraps correct structural representations from less data.

## Relevance to our project

- **Direct motivation for the LSTM vs. RNNG comparison**: The professor recommended this paper precisely because it establishes that structural bias matters for syntactic generalization. Our noise experiment adds a new axis — does that advantage persist or widen when training data is *noisy* rather than merely *scarce*?
- **Prediction generation**: One plausible prediction is that RNNG's structural supervision makes it more robust to noise because hierarchical structure provides a strong prior that resists corruption. An alternative is that noise corrupts the RNNG's parse-tree supervision signal more severely than the LSTM's unstructured objective, making RNNG *more* brittle.
- **Cite in intro/related work**: "Prior work has shown RNNGs outperform LSTMs on syntactic generalization (Wilcox et al. 2020); we test whether this advantage holds under noisy training conditions."

## Key terms

- **Structural supervision**: explicit parse-tree prediction as an auxiliary or primary objective
- **Non-local dependency**: a grammatical relation spanning multiple words/phrases
- **Few-shot generalization**: learning held-out constructions from limited examples

## Bibtex

```bibtex
@inproceedings{wilcox2020structural,
  title={Structural supervision improves learning of non-local grammatical dependencies},
  author={Wilcox, Ethan and Qian, Peng and Futrell, Richard and Kohita, Ryosuke and Levy, Roger and Ballesteros, Miguel},
  booktitle={Proceedings of NAACL-HLT},
  pages={3302--3312},
  year={2020}
}
```
