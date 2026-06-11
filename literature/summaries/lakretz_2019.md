# Lakretz et al. (2019) — The emergence of number and syntax units in LSTM language models

**Full citation:** Lakretz, Y., Kruszewski, G., Desbordes, T., Hupkes, D., Dehaene, S., & Baroni, M. (2019). The emergence of number and syntax units in LSTM language models. In *Proceedings of NAACL-HLT 2019*, 11–20.

**PDF:** https://aclanthology.org/N19-1002.pdf  
**ACL Anthology:** https://aclanthology.org/N19-1002/

---

## What the paper does

Opens up the LSTM to ask *how* it solves subject–verb agreement. Through targeted ablations and unit-level analysis on the Linzen-style task, the authors find that a **small number of specialized units** carry the agreement computation:
- Two "long-range number units" track the grammatical number of the subject across intervening material and carry it to the verb.
- Additional "syntax units" track depth of syntactic embedding, gating when the number information is read out.
- Knocking out these few units selectively destroys long-distance agreement while leaving local agreement intact.

The picture is a sparse, interpretable mechanism: the network implements something like a number register plus a syntactic controller, rather than a diffuse statistical association.

## Relevance to our project

- **Mechanistic grounding for the discussion.** Our results show the attractor gap growing under noise and the hierarchical rule failing at 50% corruption; Lakretz et al. tell us what is plausibly breaking — the long-range number/syntax circuitry that has to be learned for the hierarchical rule, versus the easy local statistics that survive.
- **Why noise hits attractor items first.** Their account predicts exactly our pattern: local (no-attractor) agreement leans on robust short-range statistics, while attractor items require the fragile long-range register that noisy supervision is least able to train.
- **Trajectory interpretation.** It frames our delay-vs-prevention finding as a question about whether those specialized units ever form — under heavy noise the model may never develop the long-range register, consistent with the 50% plateau.
- **Cite in discussion** (cognitive-modeling / how-the-rule-is-represented angle).

## Key terms

- **Number unit**: an individual LSTM cell that encodes grammatical number of the subject over long distances
- **Syntax unit**: a cell tracking syntactic embedding depth, gating the number readout
- **Mechanistic interpretability**: explaining a model's behavior in terms of identifiable internal components

## Bibtex

```bibtex
@inproceedings{lakretz2019emergence,
  title={The emergence of number and syntax units in {LSTM} language models},
  author={Lakretz, Yair and Kruszewski, German and Desbordes, Theo and Hupkes, Dieuwke and Dehaene, Stanislas and Baroni, Marco},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
  pages={11--20},
  year={2019}
}
```
