# McCoy, Frankland & Linzen (2020) — Does syntax need to grow on trees?

**Full citation:** McCoy, R. T., Frankland, S. M., & Linzen, T. (2020). Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence-to-sequence networks. In *Proceedings of CogSci 2020*.

> *Note: The README cites "McCoy, Frankland & Linzen 2020" without a title. The most likely paper is the CogSci 2020 paper above, which directly addresses hierarchical inductive biases and relates to our project's central question. There is also McCoy et al. (2020, ACL) on right-for-the-wrong-reasons generalization. If unsure, confirm with the professor.*

**PDF (CogSci 2020):** https://cogsci.mindmodeling.org/2020/papers/0039/0039.pdf  
**Alternate (ACL 2019 related):** https://aclanthology.org/P19-1334.pdf

---

## What the paper does

Asks where hierarchical inductive biases come from in sequence models, and whether they require tree-structured architectures. The paper compares:
- Standard seq2seq (LSTM encoder–decoder, no structural bias)
- Tree-structured encoders/decoders
- Span-based models

Using question formation (a syntactic transformation requiring hierarchical structure) and related tasks as test beds, the authors find that:
- Tree-structured models generalize hierarchically more reliably than flat seq2seq.
- However, even flat models can develop hierarchical representations when trained on enough data with the right distribution.
- The key variable is whether the training distribution contains enough evidence to distinguish hierarchical from linear hypotheses.

## Relevance to our project

- **Core theoretical connection**: Our project is essentially an empirical test of what McCoy et al. study theoretically: does the *input distribution* (here, corrupted by noise) determine whether a model acquires hierarchical or linear rules?
- **Poverty of stimulus angle**: The paper is part of the broader conversation about whether hierarchical structure must be built in (nativist) or can be learned from input (empiricist). Our noise manipulation directly tests the input-quality dimension.
- **Cite in intro/discussion**: When we discuss why noise might push models toward linear strategies, this paper provides the theoretical scaffolding — specifically that flat models default to linear strategies when hierarchical evidence in the input is weakened.

## Key terms

- **Hierarchical inductive bias**: a model's tendency to generalize using tree-structured rules rather than linear string operations
- **Question formation**: syntactic task requiring movement of the main-clause auxiliary (not the embedded one), a classic test of hierarchical vs. linear processing
- **Linear hypothesis**: agreement with the nearest/most recent noun, ignoring hierarchical structure

## Bibtex

```bibtex
@inproceedings{mccoy2020does,
  title={Does syntax need to grow on trees? {S}ources of hierarchical inductive bias in sequence-to-sequence networks},
  author={McCoy, R. Thomas and Frankland, Samuel M and Linzen, Tal},
  booktitle={Proceedings of the Annual Conference of the Cognitive Science Society},
  year={2020}
}
```
