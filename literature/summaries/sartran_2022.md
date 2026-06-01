# Sartran et al. (2022) — Transformer Grammars

**Full citation:** Sartran, L., Barrett, S., Vylomova, E., Noci, L., Buys, J., & Bhooshan, S. (2022). Transformer Grammars: Augmenting Transformer Language Models with Parse-Tree Annotations for Syntactic Generalization. *Transactions of the Association for Computational Linguistics*, 10, 1147–1161.

**PDF:** https://aclanthology.org/2022.tacl-1.66.pdf  
**ACL Anthology:** https://aclanthology.org/2022.tacl-1.66/

---

## What the paper does

Proposes **Transformer Grammars**, a variant of the Transformer that augments standard left-to-right language modeling with explicit parse-tree actions (analogous to the RNNG's SHIFT/NT/REDUCE operations but implemented within the Transformer attention framework). The model interleaves parse actions with word tokens in the input sequence, giving the Transformer a structural inductive bias while preserving its parallelism.

Evaluation spans:
- Standard language modeling perplexity (PTB, WikiText)
- Targeted syntactic generalization (SyntaxGym benchmark)
- Comparison against: standard Transformer LM, LSTM, and RNNG

Key results:
- Transformer Grammars outperform all baselines on targeted syntactic tests, including subject–verb agreement, while remaining competitive on perplexity.
- The structural bias improves generalization even at larger model scales.
- The paper is the contemporary state-of-the-art for syntactically-supervised language models, extending the RNNG line of work to the Transformer.

## Relevance to our project

- **Motivating the RNNG vs. LSTM comparison**: The professor's recommendation of this paper (alongside Wilcox et al. 2020) establishes the current consensus that structural bias improves syntactic generalization. Our project asks whether that advantage *changes* under noisy training.
- **Architecture framing**: If RNNG is infeasible, knowing that Transformer Grammars are the current best structural model contextualizes the fallback to ON-LSTM as a pragmatic choice rather than a principled one.
- **Cite in related work**: "Recent work has scaled syntactically-supervised models to transformers (Sartran et al. 2022), but all prior comparisons assume clean training data. Our study introduces training noise as a new experimental variable."

## Key terms

- **Transformer Grammar**: a Transformer that interleaves word tokens with explicit parse-tree action tokens
- **SyntaxGym**: the evaluation suite used throughout this paper (see also gauthier_2020.md)
- **Parse-tree annotation**: gold parse trees (from Penn Treebank or a parser) used as training supervision

## Bibtex

```bibtex
@article{sartran2022transformer,
  title={Transformer Grammars: Augmenting Transformer Language Models with Parse-Tree Annotations for Syntactic Generalization},
  author={Sartran, Laurent and Barrett, Samuel and Vylomova, Ekaterina and Noci, Lorenzo and Buys, Jan and Bhooshan, Sudeep},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={1147--1161},
  year={2022}
}
```
