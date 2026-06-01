# Gauthier et al. (2020) — SyntaxGym: An Online Platform for Targeted Evaluation of Language Models

**Full citation:** Gauthier, J., Hu, J., Wilcox, E., Qian, P., & Levy, R. (2020). SyntaxGym: An online platform for targeted evaluation of language models. In *Proceedings of ACL 2020* (pp. 70–76).

**PDF:** https://aclanthology.org/2020.acl-demos.10.pdf  
**ACL Anthology:** https://aclanthology.org/2020.acl-demos.10/  
**Platform:** http://syntaxgym.org

---

## What the paper does

Introduces **SyntaxGym**, a framework and web platform for evaluating language models on targeted syntactic phenomena using **surprisal-based test suites**. The key innovation over BLiMP-style evaluation:

- **Region-based comparisons**: rather than comparing full sentence probabilities, SyntaxGym breaks sentences into regions and compares surprisal differences at the critical region (e.g., the verb) between grammatical and ungrammatical conditions.
- **Flexible predicates**: test suites specify logical predicates (e.g., "surprisal in region 3 should be lower for the grammatical condition") that can capture complex interaction patterns.
- **Extensible**: researchers can upload new test suites; the platform evaluates any model that can produce token-level surprisals.

Test suites include subject–verb agreement (with/without attractors), reflexive anaphora, NPI licensing, center embedding, and garden-path effects.

## Relevance to our project

- **Surprisal as a metric**: Our secondary metric (surprisal of the correct verb = −log₂ P(verb | context)) comes directly from the SyntaxGym/surprisal tradition. The methodology section should cite this alongside Linzen for the surprisal measure.
- **Targeted syntactic evaluation**: SyntaxGym formalizes what we do informally — comparing model behavior between matched grammatical and ungrammatical conditions. If we need to frame our evaluation in the language of the field, SyntaxGym's region/predicate framework is the vocabulary to use.
- **Related work**: Cite when describing our evaluation methodology, especially the surprisal metric.

## Key terms

- **Surprisal**: −log₂ P(word | context); higher = more unexpected; a model that has learned SVA assigns lower surprisal to the correct verb form
- **Critical region**: the part of the sentence where the grammaticality distinction is expected to manifest (here: the verb)
- **Test suite**: a set of minimal-pair items organized around a specific grammatical phenomenon

## Bibtex

```bibtex
@inproceedings{gauthier2020syntaxgym,
  title={{SyntaxGym}: An online platform for targeted evaluation of language models},
  author={Gauthier, Jon and Hu, Jennifer and Wilcox, Ethan and Qian, Peng and Levy, Roger},
  booktitle={Proceedings of ACL},
  pages={70--76},
  year={2020}
}
```
