# Warstadt et al. (2020) — BLiMP: The Benchmark of Linguistic Minimal Pairs

**Full citation:** Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the Association for Computational Linguistics*, 8, 377–392.

**PDF:** https://aclanthology.org/2020.tacl-1.25.pdf  
**ACL Anthology:** https://aclanthology.org/2020.tacl-1.25/  
**Dataset / GitHub:** https://github.com/alexwarstadt/blimp

---

## What the paper does

Introduces **BLiMP**, a large-scale benchmark of 67,000 minimal pairs covering 67 distinct grammatical phenomena in English, spanning morphology, syntax, semantics, and binding. Each pair contrasts a grammatical and an ungrammatical sentence that differ in exactly one linguistic property. A language model "passes" a pair if it assigns higher probability to the grammatical sentence.

Key design principles:
- **Controlled generation**: pairs are generated from templates with linguistically annotated paradigms, ensuring each minimal pair isolates a single phenomenon.
- **Broad coverage**: 12 paradigm families including agreement, argument structure, NPI licensing, binding, and more.
- **Zero-shot evaluation**: no fine-tuning; the LM is evaluated solely on the probability comparison.

Key results (at time of publication):
- GPT-2 (large) achieves ~80% aggregate accuracy, with higher variance across paradigms.
- Subject–verb agreement is one of the stronger paradigms (>90% for large models), but performance drops for long-distance cases with attractors.

## Relevance to our project

- **Evaluation paradigm**: Our minimal-pair accuracy metric is conceptually identical to BLiMP's paradigm — we compare P(grammatical verb) vs. P(ungrammatical verb) given the preceding context. The difference is we use the Linzen dataset rather than BLiMP's templates, so our pairs are naturalistic rather than template-generated.
- **Related work citation**: BLiMP is the standard reference for the minimal-pair evaluation approach. Cite it in the evaluation section alongside Linzen et al. (2016) to situate our method in the broader literature.
- **Alex Warstadt's work**: Our advisor is Alex Warstadt, the first author of BLiMP. This paper directly underlies the methodological choices in our evaluation design.

## Key terms

- **Minimal pair**: two sentences differing in exactly one linguistic property, one grammatical and one not
- **Zero-shot grammaticality judgment**: using a LM's probability scores to classify grammaticality without task-specific fine-tuning
- **BLiMP paradigm**: the specific template-based construction procedure

## Bibtex

```bibtex
@article{warstadt2020blimp,
  title={{BLiMP}: The benchmark of linguistic minimal pairs for {E}nglish},
  author={Warstadt, Alex and Parrish, Alicia and Liu, Haokun and Mohananey, Anhad and Peng, Wei and Wang, Sheng-Fu and Bowman, Samuel R},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={377--392},
  year={2020}
}
```
