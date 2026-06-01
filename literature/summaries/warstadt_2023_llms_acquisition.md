# Warstadt & Bowman (2022) — What Artificial Neural Networks Can Tell Us About Human Language Acquisition

**Full citation:** Warstadt, A., & Bowman, S. R. (2022). What artificial neural networks can tell us about human language acquisition. In *Algebraic Structures in Natural Language* (pp. 17–60). CRC Press.

**Also relevant:** Warstadt, A., et al. (2023). Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. In *Proceedings of the BabyLM Challenge at CoNLL 2023*.

**arXiv (2022 chapter):** https://arxiv.org/abs/2208.07998

---

## What the paper does

A position paper / book chapter arguing that neural language models can serve as scientific models of human language acquisition, but only if used carefully. Key points:

1. **LMs as formal existence proofs**: if a neural network trained purely on text can acquire grammatical knowledge, that demonstrates the *possibility* of acquisition from distributional input — weakening the force of POS arguments without fully refuting them.
2. **The evaluation problem**: standard LM benchmarks (perplexity, downstream task accuracy) do not directly measure what is theoretically interesting — the *type* of rule acquired (hierarchical vs. linear) and the *sample efficiency* of acquisition.
3. **BabyLM framing**: proposes data-limited ("developmentally plausible") pretraining as a more ecologically valid test bed for acquisition modeling.
4. **Cautions**: argues against conflating good LM performance with human-like acquisition; highlights that LMs and children differ on the noise/negative-evidence dimension (children receive relatively clean input with no explicit negative evidence, while LMs receive vast text corpora).

## Relevance to our project

- **Theoretical grounding**: This paper, by our advisor Alex Warstadt, articulates precisely why the LM-as-cognitive-model framing of our project is scientifically legitimate and interesting. The intro and discussion sections should draw on this framing.
- **Noise as a variable**: Warstadt explicitly notes that children's input is relatively clean and that LMs trained on internet text face a different noise profile. Our study directly manipulates input noise — a variable this paper identifies as theoretically important.
- **Empiricist vs. nativist framing**: The paper provides the clearest contemporary statement of what it would mean for LMs to "address" the POS argument: not refutation, but reduction of the gap that POS was invoked to explain.
- **Cite in intro**: "We follow the program of Warstadt & Bowman (2022) in using LMs trained under controlled conditions as a tool for understanding the contributions of distributional input to grammar acquisition."

## Key terms

- **BabyLM**: the low-resource training regime (≈100M tokens) designed to be more comparable to a child's input
- **Existence proof**: demonstrating that a property *can* be learned from data, without claiming it *is* learned that way in humans
- **Developmentally plausible corpora**: training data matched to the amount/type of input children receive

## Bibtex

```bibtex
@incollection{warstadt2022artificial,
  title={What artificial neural networks can tell us about human language acquisition},
  author={Warstadt, Alex and Bowman, Samuel R},
  booktitle={Algebraic Structures in Natural Language},
  pages={17--60},
  publisher={CRC Press},
  year={2022}
}

@inproceedings{warstadt2023findings,
  title={Findings of the {BabyLM} Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora},
  author={Warstadt, Alex and others},
  booktitle={Proceedings of the BabyLM Challenge at CoNLL},
  year={2023}
}
```
